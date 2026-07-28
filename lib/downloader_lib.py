#! /bin/bash

class EventWaveformDownloaderAWS:
    def __init__(self, params):
        """
        Parameters in params
        --------------------
        aws_download_dir: str
            Parent directory for downloads. Structure: 
            aws_download_dir/{YYYY}/{MM}/{event_name}/{event_name}_{source}.ms

        eq_catalog: pandas.DataFrame
            DataFrame with required columns: ['event_name', 'edatetime', 'emag', 
            'emagtype', 'elat', 'elon', 'edep', 'nst', 'source', 'event_id']

        allowed_sample_rates: list
            List of allowed sample rates

        pre_event_time: float
            Time before event origin in seconds of desired time window

        post_event_time: float
            Time after event origin in seconds of desired time window

        max_distance: float
            Maximum distance (in km) to keep records

        STATUS: dict
            Dictionary with event status:
                FAIL      : -99
                RETRY     : -2
                PENDING   : -1
                NEW       : 0
                SUCCESS   : 1
                REDUNDANT : 2

        db_dir: str
            Directory containing database file

        source: str
            Source to download from ('scedc' to download from SCEDC)

        station_locations: dict
            Dictionary with station locations:
            station_locations[station_name] = (elon, elat, edep)

        event_origins: dict
            Dictionary with event origins:
            event_origins[event_name] = (elon, elat, edep, edatetime)
        
        
        """
        ### Initialize parameters ###

        # Parent directory for downloads. Structure: 
        #   aws_download_dir/{YYYY}/{MM}/{event_name}/{event_name}_{source}.ms
        self.aws_download_dir = params['aws_download_dir']

        # DataFrame with required columns: ['event_name', 'edatetime', 'emag', 
        # 'emagtype', 'elat', 'elon', 'edep', 'nst', 'source', 'event_id']
        self.eq_catalog = params['eq_catalog']

        self.allowed_sample_rates = params['allowed_sample_rates']
        self.pre_event_time = params['pre_event_time']
        self.post_event_time = params['post_event_time']
        self.max_distance = params['max_distance']
        self.STATUS = params['STATUS']
        self.db_dir = params['db_dir']
        self.source = params['source']
        self.event_origins = params['event_origins']
        self.station_locations = params['station_locations']
        

        # Initialize new columns
        self.eq_catalog['status'] = 0

        self.table_name = 'event_waveform_status'
        self.db_path = f"{self.db_dir}/{self.table_name}.db"

        self.db_columns = ['event_id', 'origin_time', 'emag', 'elat', 'elon', 'edep', 'source', 'status']

    def _set_source(self):
        if self.source.upper() == 'SCEDC':
            self.source = 'SCEDC'
            self.bucket = 'scedc-pds'

            self.eq_catalog = self.eq_catalog[self.eq_catalog['source'] == 's']
            self.eq_catalog.reset_index(inplace=True, drop=True)

        else:
            print("Invalid source, or not implemented.")
            raise ValueError()
        print(f"Source set to: {self.source}. Bucket: {self.bucket}")
        self.nevents = len(self.eq_catalog)


    def _create_database(self):

        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()

        cur.execute(f"""
            CREATE TABLE {self.table_name} (
                event_id INTEGER PRIMARY KEY,
                origin_time INTEGER,
                emag FLOAT,
                elat FLOAT,
                elon FLOAT,
                edep FLOAT,
                source TEXT,
                status INTEGER
            )
        """)

        cur.executemany(f"""
            INSERT INTO {self.table_name} ({', '.join(self.db_columns)}) VALUES ({', '.join(['?'] * len(self.db_columns))})
        """, list(self.eq_catalog[self.db_columns].values))


        self.conn.commit()


    def _update_status(self):
        self._check_downloaded_files()
        self._print_status_counts()
    
    def _get_scedc_bucket_key_filename_event(self, event_id, date):
        # date is a UTCDateTime object
        # Tested, working
        # example key: event_waveforms/2022/2022_001/38438519.ms
        julday = str(date.julday).zfill(3)
        year = date.year
        
        prefix = f"event_waveforms/{year}/{year}_{julday}/"
        filename = f"{event_id}.ms"
        key = prefix + filename

        return key, filename

    def _clean_stream(self, st_downloaded, event_origin):
        """
        input a downloaded Obspy stream
        Perform the following actions:
        1) Remove traces farther than max_distance
        2) Remove traces with weird sampling rates
        3) Slice stream to desired time window

        event_origin should be (elon, elat, edep, edatetime)
        easily accessed in lookup dict: event_origins[event_name]

        need 'from obspy import Stream'

        """

        elon, elat, edep, edatetime = event_origin

        t1 = edatetime - self.pre_event_time
        t2 = edatetime + self.post_event_time

        st_clean = Stream()

        ntr = len(st_downloaded)
        dist = [[]] * ntr

        # 1) Filter out far traces
        for i, tr in enumerate(st_downloaded):
            station_name = '.'.join(tr.get_id().split('.')[:2])
            slon, slat, _ = self.station_locations[station_name]
            dist[i] = haversine_np(elon, elat, slon, slat)
            st_downloaded[i].stats.distance = dist[i] * 1E3
            if dist[i] <= self.max_distance:
                st_clean.append(tr.copy())
        
        # 2) Remove traces with weird sampling rates
        st_clean = Stream(
            [tr for tr in st_clean if tr.stats.sampling_rate in self.allowed_sample_rates]
        )

        # 3) Slice stream to desired time window
        st_clean.trim(t1, t2)

        return st_clean



    def _download_aws_event(self, event_id, date):
        key, filename = self._get_scedc_bucket_key_filename_event(event_id, date)
        year, month = date.year, str(date.month).zfill(2)
        filedir = f"{self.aws_download_dir}{year}/{month}/"
        os.makedirs(filedir, exist_ok=True)

        event_name = self.source[0].lower() + str(event_id)

        download_filepath = os.path.join(filedir, filename)
        clean_filepath = os.path.join(filedir, f"{event_name}/{event_name}_{source.lower()}.ms")
        # need error handling here
        try:
            if not os.path.exists(download_filepath):
                self.boto3_res.Bucket(self.bucket).download_file(
                    key, 
                    download_filepath
                )
                print(f'downloaded {event_id}')
                self._set_event_status(event_id, 'downloaded')
            elif os.path.exists(download_filepath):
                # if file exists...
                st_downloaded = obspy.read(download_filepath)
                
                st_clean = self._clean_stream(st_downloaded, self.event_origins[event_name])

                # make clean dir
                os.makedirs(os.path.dirname(clean_filepath), exist_ok=True)
                st_clean.write(clean_filepath, format='MSEED')

                self._set_event_status(event_id, 'processed')
                # delete downloaded file
                os.remove(download_filepath)
                
        
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'LimitExceededException':
                raise e
            
            elif '404' in str(e):
                # print(f"Error downloading {event_id}: {str(e)}")
                self._set_event_status(event_id, 'failed')
            else:
                raise e
        
        except Exception as e:
            print(f"Error downloading {event_id}: {e}")
            print(type(e))
            raise e
        

    
    def _print_status_counts(self):
        print("Dataset status:")
        print("----------------")
        for key, value in self.STATUS.items():
            count = len(self.eq_df[self.eq_df['status'] == value])
            print(f"{key:<15}: {count:>10,}")
    
    def _set_event_status(self, event_id, status_name):
        cur = self.conn.cursor()
        cur.execute(f"""
            UPDATE {self.table_name}
            SET status = ?
            WHERE event_id = ?
        """, (self.STATUS[status_name], event_id))
        self.conn.commit()


    def _check_downloaded_files(self):
        cur = self.conn.cursor()
        for i in trange(self.nevents, desc="Checking downloaded files"):
            edatetime = self.eq_df['edatetime'].values[i]
            event_id = self.eq_df['event_id'].values[i]

            year, month = edatetime.year, edatetime.month

            filepath = f"{self.aws_download_dir}{year}/{month:02}/{event_id}.ms"
            # print(filepath)
            if os.path.exists(filepath):
                self.eq_df.loc[i, 'status'] = self.STATUS['downloaded']
                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET status = ?
                    WHERE event_id = ?
                """, (self.STATUS['downloaded'], event_id))
        self.conn.commit()



    def _init_boto3(self):
        # Initialize boto3 downloader
        self.boto3_res = boto3.resource(
            's3', 
            config=Config(signature_version=UNSIGNED)
        )

    def run_downloader(self):
        self._set_source()

        try:
            # if database doesn't exist...
            if not os.path.exists(self.db_path):
                print("Database not found. Creating new database...")
                # create origin_time int column
                self.eq_catalog['origin_time'] = [dt.ns for dt in self.eq_catalog['edatetime']]
                self.eq_df = self.eq_catalog[self.db_columns].copy()

                # create database using eq_df data
                # Database connection opened
                self._create_database()
                # Database connection closed
            else:
                self.conn = sqlite3.connect(self.db_path)
            
            self.eq_df = pd.read_sql(f"SELECT * FROM {self.table_name}", self.conn)
            
            self.eq_df['edatetime'] = [UTCDateTime(ns=t) for t in self.eq_df['origin_time']]

            self._update_status()
            self._init_boto3()

            for i in trange(self.nevents, desc="Downloading waveforms"):
                edatetime = self.eq_df['edatetime'].values[i]
                event_id = self.eq_df['event_id'].values[i]
                status_value = self.eq_df['status'].values[i]
                if status_value == self.STATUS['new']: # change this to 'new' later!!!
                    self._download_aws_event(event_id, edatetime)
        finally:
            self.conn.close()