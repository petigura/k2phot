CREATE TABLE star (
  id INTEGER PRIMARY KEY,
  starname TEXT,
  weights_count int,
  weights_min REAL,
  weights_max REAL,
  weights_std REAL,
  pix_basedir TEXT,
  pix_filename TEXT,
  phot_basedir TEXT,
  phot_filename TEXT
);
