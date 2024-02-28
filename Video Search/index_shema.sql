CREATE EXTENSION IF NOT EXISTS pgvector;

CREATE TABLE videos (
    vid_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    upload_date DATE NOT NULL
);


CREATE TABLE segments (
    segment_id SERIAL PRIMARY KEY,
    vid_id INTEGER NOT NULL REFERENCES videos(vid_id),
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    description TEXT,
    start_frame_id INTEGER,  -- key frame
    end_frame_id INTEGER
);


CREATE TABLE frames (
    frame_id SERIAL PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments(segment_id),
    vid_id INTEGER NOT NULL REFERENCES videos(vid_id),
    frame_num INTEGER NOT NULL,
    timestamp FLOAT NOT NULL,
    is_key_frame BOOLEAN DEFAULT FALSE -- key frame or not
);


CREATE TABLE objects (
    frame_id INTEGER NOT NULL,
    vid_id INTEGER NOT NULL,
    detected_obj_class TEXT NOT NULL,
    detectedObjId TEXT NOT NULL,
    bbox_info TEXT NOT NULL,
    confidence FLOAT,
    vector VECTOR(32),
    PRIMARY KEY (frame_id, vid_id, detected_obj_class, bbox_info, detectedObjId), -- 更新主键定义
    FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
    FOREIGN KEY (vid_id) REFERENCES videos(vid_id)
);


CREATE INDEX idx_object_vector ON objects USING ivfflat (vector);
