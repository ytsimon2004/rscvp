BEGIN TRANSACTION;

-- Create new table with modified schema
CREATE TABLE FieldOfViewDB_new
(
    date                    TEXT    NOT NULL,
    animal                  TEXT    NOT NULL,
    user                    TEXT    NOT NULL,
    usage                   TEXT    NOT NULL,
    region                  TEXT    NOT NULL,
    max_depth               TEXT    NOT NULL,
    n_planes                INTEGER NOT NULL,
    objective_rotation      FLOAT   NOT NULL,
    objective_magnification INTEGER NOT NULL,

    medial_anterior         TEXT,
    medial_posterior        TEXT,
    lateral_posterior       TEXT,
    lateral_anterior        TEXT,

    PRIMARY KEY (date, animal, user),
    FOREIGN KEY (date, animal, user)
        REFERENCES PhysiologyDB (date, animal, user)
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
);

-- Insert old table to new table
INSERT INTO FieldOfViewDB_new
SELECT *
FROM FieldOfViewDB;


-- Remove old
DROP TABLE FieldOfViewDB;

-- Rename new
ALTER TABLE FieldOfViewDB_new
    RENAME TO FieldOfViewDB;

COMMIT;
