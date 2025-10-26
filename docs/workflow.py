from pathlib import Path

from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.programming.flowchart import StartEnd, Action, Database, StoredData
from diagrams.programming.language import Python, Sql, Matlab


def physiology_workflow():
    with Diagram("Physiological Data Analysis Workflow", direction="LR"):
        physiology = StartEnd("Physiology Data")

        with Cluster("Raw Data Type"):
            behavior = Database("Behavior\n(running, pupil, facial,...)")
            wide_field = Database("Wide Field\n(retinotopy)")
            cellular = Database("2P Calcium\n(cellular)")

        with Cluster("Behavioral Processing"):
            behavioral_session = Action("Session")
            behavioral_batch = Action("Batch")
            behavioral_module = Python("behavioral/\ntrack")

        with Cluster("Wide-field Processing"):
            widefield_module = Python("retinotopic/model")

        with Cluster("Cellular Processing"):
            cellular_session = Action("Session-level")
            cellular_batch = Action("Batch-level")

        with Cluster("Session Analysis"):
            csv_out = StoredData("metrics.csv")
            cache_out = StoredData("persistence.pkl")
            db_session = Sql("SQLite DB")
            cellular_modules = Python("selection/visual\nspatial/topology\n/model")

        with Cluster("Batch Statistics"):
            parquet_out = StoredData("csv aggregate (.parquet)")
            cache_agg = StoredData("persistence aggregate (.pkl)")
            db_batch = Sql("SQLite DB")
            statistic_modules = Python("statistic\n")

        # Physiology
        physiology >> [behavior, wide_field, cellular]

        # Behavioral flow
        behavior >> behavioral_session >> behavioral_module
        behavior >> behavioral_batch >> behavioral_module

        # Wide-field flow
        wide_field >> widefield_module

        # Cellular session flow
        cellular >> cellular_session >> [csv_out, cache_out, db_session]
        [csv_out, cache_out, db_session] >> cellular_modules

        # Cellular batch flow
        cellular >> cellular_batch >> [parquet_out, cache_agg, db_batch]
        [parquet_out, cache_agg, db_batch] >> statistic_modules


def histology_workflow():
    with Diagram("Histology Data Analysis Workflow", direction="LR"):
        histology = StartEnd("Histology Data")
        image_processing = Action("Image processing")

        with Cluster("Annotation/Segmentation"):
            imagej_logo = str(Path(__file__).parent / 'source/_static/imageJ_logo.png')
            annotation = Custom("ImageJ\nmanual annotation", imagej_logo)
            segmentation = Python("cellpose/stardist\nautomated segmentation")

        with Cluster("Registration"):
            registration = Matlab("CCF registration\n(.csv output)")

        atlas_module = Python("atlas")

        # Define connections
        histology >> image_processing >> [annotation, segmentation]
        annotation >> registration
        segmentation >> registration
        registration >> atlas_module


histology_workflow()
