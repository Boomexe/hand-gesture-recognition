TO DO STUFF:

- get rid of "warming up" phase by populating entire list with zeros instead of slowly adding frames one by one until we reach a length of 93
(so like just begin with having 93 frames of zeros only)

- get rid of delay between gestures by instead of waiting for empty frames to push out the old frames, keep a count of how many empty frames were sent in a row.
  if maybe 10-20 or so frames are detected in a row as "empty," then we can discard the current data cache and populate with zeros.