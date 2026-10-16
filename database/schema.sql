-- SonicSentinel AI database schema (generated from src/models.py)

CREATE TABLE app_settings (
	"key" VARCHAR(60) NOT NULL, 
	value JSON, 
	updated_at DATETIME, 
	updated_by VARCHAR(40), 
	PRIMARY KEY ("key")
);

CREATE TABLE id_counters (
	"key" VARCHAR(30) NOT NULL, 
	value INTEGER NOT NULL, 
	PRIMARY KEY ("key")
);

CREATE TABLE model_versions (
	id INTEGER NOT NULL, 
	model_type VARCHAR(10) NOT NULL, 
	version VARCHAR(120) NOT NULL, 
	algorithm VARCHAR(80), 
	source VARCHAR(400), 
	labels JSON, 
	metrics JSON, 
	is_active BOOLEAN, 
	registered_at DATETIME, 
	registered_by VARCHAR(40), 
	notes TEXT, 
	PRIMARY KEY (id), 
	UNIQUE (model_type, version)
);

CREATE TABLE system_alerts (
	id INTEGER NOT NULL, 
	kind VARCHAR(40) NOT NULL, 
	severity VARCHAR(16), 
	message VARCHAR(300) NOT NULL, 
	details JSON, 
	count INTEGER, 
	resolved BOOLEAN, 
	resolved_by VARCHAR(40), 
	resolved_at DATETIME, 
	created_at DATETIME, 
	last_seen_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_system_alerts_resolved ON system_alerts (resolved);
CREATE INDEX ix_system_alerts_kind ON system_alerts (kind);

CREATE TABLE users (
	id INTEGER NOT NULL, 
	user_code VARCHAR(16) NOT NULL, 
	username VARCHAR(40) NOT NULL, 
	email VARCHAR(120) NOT NULL, 
	password_hash VARCHAR(255) NOT NULL, 
	full_name VARCHAR(120) NOT NULL, 
	role VARCHAR(20) NOT NULL, 
	phone VARCHAR(30), 
	organization VARCHAR(120), 
	department VARCHAR(120), 
	job_title VARCHAR(120), 
	active_account BOOLEAN NOT NULL, 
	mic_consent_at DATETIME, 
	preferences JSON, 
	failed_logins INTEGER NOT NULL, 
	locked_until DATETIME, 
	last_login_at DATETIME, 
	created_at DATETIME NOT NULL, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (email)
);
CREATE UNIQUE INDEX ix_users_user_code ON users (user_code);
CREATE UNIQUE INDEX ix_users_username ON users (username);

CREATE TABLE audit_logs (
	id INTEGER NOT NULL, 
	user_id INTEGER, 
	username VARCHAR(40), 
	action VARCHAR(40) NOT NULL, 
	target_type VARCHAR(30), 
	target_id VARCHAR(40), 
	details JSON, 
	ip_address VARCHAR(64), 
	user_agent VARCHAR(200), 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_audit_logs_created_at ON audit_logs (created_at);
CREATE INDEX ix_audit_logs_action ON audit_logs (action);
CREATE INDEX ix_audit_logs_user_id ON audit_logs (user_id);

CREATE TABLE batches (
	id INTEGER NOT NULL, 
	batch_code VARCHAR(24) NOT NULL, 
	user_id INTEGER NOT NULL, 
	file_count INTEGER, 
	processed INTEGER, 
	failed INTEGER, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (batch_code), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE evaluation_runs (
	id INTEGER NOT NULL, 
	run_code VARCHAR(24) NOT NULL, 
	created_by INTEGER, 
	status VARCHAR(16), 
	per_class INTEGER, 
	total INTEGER, 
	completed INTEGER, 
	python_version VARCHAR(64), 
	gtm_version VARCHAR(120), 
	summary JSON, 
	created_at DATETIME, 
	finished_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (run_code), 
	FOREIGN KEY(created_by) REFERENCES users (id)
);

CREATE TABLE mic_sessions (
	id INTEGER NOT NULL, 
	session_code VARCHAR(24) NOT NULL, 
	user_id INTEGER NOT NULL, 
	status VARCHAR(16), 
	window_seconds FLOAT, 
	hop_seconds FLOAT, 
	device_label VARCHAR(200), 
	location VARCHAR(120), 
	windows_processed INTEGER, 
	silent_windows INTEGER, 
	detections INTEGER, 
	alerts INTEGER, 
	avg_latency_ms FLOAT, 
	started_at DATETIME, 
	last_window_at DATETIME, 
	ended_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (session_code), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_mic_sessions_user_id ON mic_sessions (user_id);

CREATE TABLE audio_records (
	id INTEGER NOT NULL, 
	audio_id VARCHAR(24) NOT NULL, 
	user_id INTEGER NOT NULL, 
	source VARCHAR(12), 
	batch_id INTEGER, 
	session_id INTEGER, 
	evaluation_run_id INTEGER, 
	window_index INTEGER, 
	dataset_audio_id VARCHAR(32), 
	actual_class VARCHAR(40), 
	original_filename VARCHAR(255), 
	stored_path VARCHAR(400), 
	processed_path VARCHAR(400), 
	spectrogram_path VARCHAR(400), 
	waveform_path VARCHAR(400), 
	audio_deleted BOOLEAN, 
	format VARCHAR(16), 
	codec VARCHAR(32), 
	file_size INTEGER, 
	duration FLOAT, 
	sample_rate INTEGER, 
	channels INTEGER, 
	bit_depth INTEGER, 
	sha256 VARCHAR(64), 
	fingerprint BLOB, 
	fp_descriptor BLOB, 
	duplicate_of_id INTEGER, 
	near_duplicate_of_id INTEGER, 
	near_duplicate_ber FLOAT, 
	validation JSON, 
	metadata_extra JSON, 
	preprocessing JSON, 
	quality_label VARCHAR(12), 
	quality_score INTEGER, 
	quality JSON, 
	status VARCHAR(20), 
	gtm_status VARCHAR(12), 
	python_top VARCHAR(40), 
	python_confidence FLOAT, 
	python_scores JSON, 
	python_margin FLOAT, 
	gtm_top VARCHAR(40), 
	gtm_confidence FLOAT, 
	gtm_scores JSON, 
	gtm_margin FLOAT, 
	class_match BOOLEAN, 
	confidence_difference FLOAT, 
	margin_difference FLOAT, 
	consistency VARCHAR(24), 
	final_category VARCHAR(40), 
	final_confidence FLOAT, 
	confidence_level VARCHAR(8), 
	severity VARCHAR(16), 
	alert_status VARCHAR(24), 
	needs_review BOOLEAN, 
	review_reasons JSON, 
	uncertain_reasons JSON, 
	overlapping BOOLEAN, 
	overlap_classes JSON, 
	recommended_action TEXT, 
	decision JSON, 
	events JSON, 
	python_model_version VARCHAR(64), 
	gtm_model_version VARCHAR(120), 
	reviewed_category VARCHAR(40), 
	reviewed_severity VARCHAR(16), 
	reviewed_by INTEGER, 
	reviewed_at DATETIME, 
	error_message TEXT, 
	processing_ms INTEGER, 
	is_evaluation BOOLEAN, 
	created_at DATETIME, 
	classified_at DATETIME, 
	closed_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(batch_id) REFERENCES batches (id), 
	FOREIGN KEY(session_id) REFERENCES mic_sessions (id), 
	FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs (id), 
	FOREIGN KEY(duplicate_of_id) REFERENCES audio_records (id), 
	FOREIGN KEY(near_duplicate_of_id) REFERENCES audio_records (id), 
	FOREIGN KEY(reviewed_by) REFERENCES users (id)
);
CREATE INDEX ix_audio_records_consistency ON audio_records (consistency);
CREATE UNIQUE INDEX ix_audio_records_audio_id ON audio_records (audio_id);
CREATE INDEX ix_audio_records_final_confidence ON audio_records (final_confidence);
CREATE INDEX ix_audio_records_final_category ON audio_records (final_category);
CREATE INDEX ix_records_created_category ON audio_records (created_at, final_category);
CREATE INDEX ix_audio_records_user_id ON audio_records (user_id);
CREATE INDEX ix_audio_records_session_id ON audio_records (session_id);
CREATE INDEX ix_audio_records_needs_review ON audio_records (needs_review);
CREATE INDEX ix_audio_records_severity ON audio_records (severity);
CREATE INDEX ix_audio_records_status ON audio_records (status);
CREATE INDEX ix_audio_records_sha256 ON audio_records (sha256);
CREATE INDEX ix_audio_records_quality_label ON audio_records (quality_label);
CREATE INDEX ix_audio_records_evaluation_run_id ON audio_records (evaluation_run_id);
CREATE INDEX ix_audio_records_created_at ON audio_records (created_at);
CREATE INDEX ix_audio_records_is_evaluation ON audio_records (is_evaluation);

CREATE TABLE alerts (
	id INTEGER NOT NULL, 
	alert_code VARCHAR(24) NOT NULL, 
	record_id INTEGER NOT NULL, 
	session_id INTEGER, 
	segment_index INTEGER, 
	category VARCHAR(40) NOT NULL, 
	severity VARCHAR(16) NOT NULL, 
	alert_type VARCHAR(20), 
	source VARCHAR(12), 
	confidence FLOAT, 
	consistency VARCHAR(24), 
	message VARCHAR(300), 
	recommended_action TEXT, 
	conditions JSON, 
	responsible_roles JSON, 
	status VARCHAR(16), 
	auto_escalated BOOLEAN, 
	escalated_to VARCHAR(160), 
	occurrences INTEGER, 
	created_at DATETIME, 
	last_seen_at DATETIME, 
	first_action_at DATETIME, 
	resolved_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (alert_code), 
	FOREIGN KEY(record_id) REFERENCES audio_records (id), 
	FOREIGN KEY(session_id) REFERENCES mic_sessions (id)
);
CREATE INDEX ix_alerts_record_id ON alerts (record_id);
CREATE INDEX ix_alerts_severity ON alerts (severity);
CREATE INDEX ix_alerts_status ON alerts (status);
CREATE INDEX ix_alerts_category ON alerts (category);
CREATE INDEX ix_alerts_created_at ON alerts (created_at);

CREATE TABLE reviews (
	id INTEGER NOT NULL, 
	record_id INTEGER NOT NULL, 
	reviewer_id INTEGER NOT NULL, 
	decision VARCHAR(16) NOT NULL, 
	original_category VARCHAR(40), 
	reviewed_category VARCHAR(40), 
	reviewed_severity VARCHAR(16), 
	is_override BOOLEAN, 
	comment TEXT, 
	recommended_action TEXT, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(record_id) REFERENCES audio_records (id), 
	FOREIGN KEY(reviewer_id) REFERENCES users (id)
);
CREATE INDEX ix_reviews_record_id ON reviews (record_id);

CREATE TABLE segments (
	id INTEGER NOT NULL, 
	record_id INTEGER NOT NULL, 
	"index" INTEGER NOT NULL, 
	start_time FLOAT NOT NULL, 
	end_time FLOAT NOT NULL, 
	proc_start FLOAT NOT NULL, 
	proc_end FLOAT NOT NULL, 
	rms_dbfs FLOAT, 
	is_event BOOLEAN, 
	silent BOOLEAN, 
	quality_label VARCHAR(12), 
	quality JSON, 
	python_scores JSON, 
	python_top VARCHAR(40), 
	python_confidence FLOAT, 
	gtm_scores JSON, 
	gtm_top VARCHAR(40), 
	gtm_confidence FLOAT, 
	gtm_window_start FLOAT, 
	final_category VARCHAR(40), 
	final_confidence FLOAT, 
	consistency VARCHAR(24), 
	severity VARCHAR(16), 
	alert_status VARCHAR(24), 
	needs_review BOOLEAN, 
	decision JSON, 
	PRIMARY KEY (id), 
	FOREIGN KEY(record_id) REFERENCES audio_records (id)
);
CREATE INDEX ix_segments_record_id ON segments (record_id);

CREATE TABLE alert_actions (
	id INTEGER NOT NULL, 
	alert_id INTEGER NOT NULL, 
	user_id INTEGER, 
	action VARCHAR(16) NOT NULL, 
	comment TEXT, 
	escalate_to VARCHAR(160), 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(alert_id) REFERENCES alerts (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_alert_actions_alert_id ON alert_actions (alert_id);
