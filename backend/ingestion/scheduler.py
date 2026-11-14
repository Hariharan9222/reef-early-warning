"""
backend/ingestion/scheduler.py
Background task scheduler using APScheduler.
Automates daily NOAA ERDDAP SST polling and weekly feature aggregations.
"""

import logging
from datetime import date, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from backend.config import settings

logger = logging.getLogger("coralguard.scheduler")

scheduler = BackgroundScheduler()

def scheduled_sst_ingestion():
    """Daily automated job fetching NOAA Coral Reef Watch SST for all monitored sites."""
    logger.info("Executing scheduled NOAA ERDDAP SST ingestion task...")
    try:
        from backend.database import SessionLocal
        from backend.models.db_models import ReefSite, SSTDaily
        from backend.ingestion.erddap_client import NOAAERDDAPClient

        db = SessionLocal()
        erddap = NOAAERDDAPClient()
        sites = db.query(ReefSite).all()
        target_date = date.today() - timedelta(days=1)  # Yesterday's satellite snapshot

        for site in sites:
            # Check if already present
            existing = db.query(SSTDaily).filter_by(site_id=site.id, date=target_date).first()
            if not existing:
                res = erddap.fetch_daily_sst_point(site.latitude, site.longitude, target_date)
                record = SSTDaily(
                    site_id=site.id,
                    date=target_date,
                    sst=res.get("sst"),
                    ssta=res.get("ssta"),
                    dhw=res.get("dhw"),
                    bleaching_alert_level=res.get("bleaching_alert_level", 0),
                    status=res.get("status", "no_data")
                )
                db.add(record)
        db.commit()
        db.close()
        logger.info(f"Scheduled SST ingestion complete for {len(sites)} sites on {target_date}.")
    except Exception as e:
        logger.error(f"Error in scheduled SST ingestion: {e}")

def init_scheduler():
    """Initializes and starts the APScheduler background worker."""
    if not settings.SCHEDULER_ENABLED:
        logger.info("APScheduler is disabled by configuration (SCHEDULER_ENABLED=False).")
        return

    if not scheduler.running:
        # Schedule daily SST fetch at specified UTC hour
        scheduler.add_job(
            scheduled_sst_ingestion,
            trigger="cron",
            hour=settings.INGESTION_CRON_HOUR,
            minute=0,
            id="daily_noaa_sst_fetch",
            replace_existing=True
        )
        scheduler.start()
        logger.info(f"APScheduler started. Daily NOAA SST fetch scheduled at {settings.INGESTION_CRON_HOUR:02d}:00 UTC.")

def shutdown_scheduler():
    """Stops the scheduler on application shutdown."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")

