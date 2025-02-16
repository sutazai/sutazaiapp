import loggingfrom logging.handlers import RotatingFileHandlerimport osimport sentry_sdkfrom sentry_sdk.integrations.logging import LoggingIntegrationdef setup_logging():    logging.basicConfig(        level = (logging.INFO),        format = ('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),        handlers = ([            RotatingFileHandler('app.log'), maxBytes = (10485760), backupCount = (5)),            logging.StreamHandler()        ]    )        # Add Sentry for error tracking    sentry_sdk.init(        dsn = (os.getenv('SENTRY_DSN')),        integrations=[LoggingIntegration()]    ) 