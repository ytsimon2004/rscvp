-- ref: https://www.sqlitetutorial.net/sqlite-alter-table/
-- disable foreign key constraint check
PRAGMA
foreign_keys=off;

-- start a transaction
BEGIN
TRANSACTION;

-- Here you can drop column
CREATE TABLE IF NOT EXISTS GenericDBNew
(
    date
    TEXT
    NOT
    NULL,
    animal
    TEXT
    NOT
    NULL,
    rec
    TEXT
    NOT
    NULL,
    user
    TEXT
    NOT
    NULL,
    optic
    TEXT
    NOT
    NULL,
    n_total_neurons
    INT
    DEFAULT
    NULL,
    n_selected_neurons
    INT
    DEFAULT
    NULL,
    n_visual_neurons
    INT
    DEFAULT
    NULL,
    n_spatial_neurons
    INT
    DEFAULT
    NULL,
    n_overlap_neuron
    INT
    DEFAULT
    NULL,
    update_time
    DATETIME
    DEFAULT
    NULL,
    PRIMARY
    KEY
(
    date,
    animal,
    rec,
    user,
    optic
) ,
    FOREIGN KEY
(
    date,
    animal,
    rec,
    user,
    optic
) REFERENCES CalImageDB
(
    date,
    animal,
    rec,
    user,
    optic
)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION
    );
-- copy data from the table to the new_table
INSERT INTO GenericDBNew(date, animal, rec, user, optic, n_total_neurons, n_selected_neurons, n_visual_neurons,
                         n_spatial_neurons, update_time)
SELECT *
FROM GenericDB;

-- drop the table
DROP TABLE GenericDB;

-- rename the new_table to the table
ALTER TABLE GenericDBNew RENAME TO GenericDB;

-- commit the transaction
COMMIT;

-- enable foreign key constraint check
PRAGMA
foreign_keys=on;