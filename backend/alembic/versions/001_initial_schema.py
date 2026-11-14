"""Initial schema for CoralGuard multimodal bleaching system.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-11-13 18:38:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. sites
    op.create_table(
        'sites',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('region', sa.String(length=100), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('depth_meters', sa.Float(), server_default='10.0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_sites_id', 'sites', ['id'])
    op.create_index('ix_sites_region', 'sites', ['region'])
    op.create_index('idx_site_coords', 'sites', ['latitude', 'longitude'])

    # 2. sst_daily
    op.create_table(
        'sst_daily',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('site_id', sa.String(length=100), sa.ForeignKey('sites.id'), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('sst', sa.Float(), nullable=True),
        sa.Column('ssta', sa.Float(), nullable=True),
        sa.Column('dhw', sa.Float(), nullable=True),
        sa.Column('bleaching_alert_level', sa.Integer(), server_default='0'),
        sa.Column('status', sa.String(length=50), server_default='verified'),
        sa.Column('fetched_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_sst_daily_site_id', 'sst_daily', ['site_id'])
    op.create_index('ix_sst_daily_date', 'sst_daily', ['date'])
    op.create_index('idx_site_date', 'sst_daily', ['site_id', 'date'], unique=True)

    # 3. acoustic_recordings
    op.create_table(
        'acoustic_recordings',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('site_id', sa.String(length=100), sa.ForeignKey('sites.id'), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('duration_sec', sa.Float(), server_default='5.0'),
        sa.Column('sample_rate', sa.Integer(), server_default='22050'),
        sa.Column('hydrophone_type', sa.String(length=100), server_default='SoundTrap 300HF'),
        sa.Column('aci', sa.Float(), nullable=True),
        sa.Column('adi', sa.Float(), nullable=True),
        sa.Column('spectral_entropy', sa.Float(), nullable=True),
        sa.Column('bioacoustic_index', sa.Float(), nullable=True),
        sa.Column('ndsi', sa.Float(), nullable=True),
        sa.Column('shrimp_band_energy_ratio', sa.Float(), nullable=True),
        sa.Column('fish_chorus_energy_ratio', sa.Float(), nullable=True),
        sa.Column('spectral_centroid_mean', sa.Float(), nullable=True),
        sa.Column('mfccs_json', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default='verified'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_acoustic_site_id', 'acoustic_recordings', ['site_id'])
    op.create_index('ix_acoustic_recorded_at', 'acoustic_recordings', ['recorded_at'])

    # 4. citizen_reports
    op.create_table(
        'citizen_reports',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('site_id', sa.String(length=100), sa.ForeignKey('sites.id'), nullable=True),
        sa.Column('observed_at', sa.Date(), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('observer_id', sa.String(length=100), nullable=False),
        sa.Column('observer_type', sa.String(length=100), server_default='recreational_diver'),
        sa.Column('survey_program', sa.String(length=100), server_default='Direct Submission'),
        sa.Column('depth_meters', sa.Float(), server_default='8.0'),
        sa.Column('bleaching_percent', sa.Float(), nullable=False),
        sa.Column('has_photo', sa.Boolean(), server_default='0'),
        sa.Column('photo_url', sa.String(length=500), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('trust_score', sa.Float(), server_default='0.5'),
        sa.Column('status', sa.String(length=50), server_default='verified'),
        sa.Column('submitted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_citizen_site_id', 'citizen_reports', ['site_id'])
    op.create_index('ix_citizen_observed_at', 'citizen_reports', ['observed_at'])
    op.create_index('idx_citizen_coords', 'citizen_reports', ['latitude', 'longitude'])

    # 5. weekly_features
    op.create_table(
        'weekly_features',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('site_id', sa.String(length=100), sa.ForeignKey('sites.id'), nullable=False),
        sa.Column('week_start_date', sa.Date(), nullable=False),
        sa.Column('sst', sa.Float(), nullable=True),
        sa.Column('ssta', sa.Float(), nullable=True),
        sa.Column('dhw', sa.Float(), nullable=True),
        sa.Column('ssta_roll_7d', sa.Float(), nullable=True),
        sa.Column('ssta_roll_14d', sa.Float(), nullable=True),
        sa.Column('ssta_roll_30d', sa.Float(), nullable=True),
        sa.Column('sst_max_30d', sa.Float(), nullable=True),
        sa.Column('acoustic_is_missing', sa.Integer(), server_default='1'),
        sa.Column('shrimp_band_energy_ratio', sa.Float(), nullable=True),
        sa.Column('fish_chorus_energy_ratio', sa.Float(), nullable=True),
        sa.Column('aci', sa.Float(), nullable=True),
        sa.Column('adi', sa.Float(), nullable=True),
        sa.Column('bioacoustic_index', sa.Float(), nullable=True),
        sa.Column('ndsi', sa.Float(), nullable=True),
        sa.Column('spectral_entropy', sa.Float(), nullable=True),
        sa.Column('diver_reports_count_50km_30d', sa.Float(), server_default='0.0'),
        sa.Column('diver_weighted_severity_30d', sa.Float(), server_default='0.0'),
        sa.Column('diver_mean_confidence_30d', sa.Float(), server_default='0.0'),
        sa.Column('diver_has_reports', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_features_site_id', 'weekly_features', ['site_id'])
    op.create_index('ix_features_week_start', 'weekly_features', ['week_start_date'])
    op.create_index('idx_site_week', 'weekly_features', ['site_id', 'week_start_date'], unique=True)

    # 6. bleaching_predictions
    op.create_table(
        'bleaching_predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('site_id', sa.String(length=100), sa.ForeignKey('sites.id'), nullable=False),
        sa.Column('forecast_date', sa.Date(), nullable=False),
        sa.Column('lead_days', sa.Integer(), nullable=False),
        sa.Column('p_none', sa.Float(), nullable=False),
        sa.Column('p_mild', sa.Float(), nullable=False),
        sa.Column('p_severe', sa.Float(), nullable=False),
        sa.Column('p_bleach', sa.Float(), nullable=False),
        sa.Column('alert_tier', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=50), server_default='xgb_v2'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_predictions_site_id', 'bleaching_predictions', ['site_id'])
    op.create_index('ix_predictions_forecast_date', 'bleaching_predictions', ['forecast_date'])


def downgrade() -> None:
    op.drop_table('bleaching_predictions')
    op.drop_table('weekly_features')
    op.drop_table('citizen_reports')
    op.drop_table('acoustic_recordings')
    op.drop_table('sst_daily')
    op.drop_table('sites')

