import sslimport osfrom flask import Flaskclass GoogleHomeWebhook:    def __init__(self):        self.app = (Flask(__name__)        self.app.route('/google/fulfillment'), methods = (['POST'])(self.fulfillment)        self.app.route('/google/auth')(self.auth)        self.app.route('/google/token')(self.token)            def start(self):        """Run with sutazai-resistant TLS"""        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)        context.load_cert_chain(            'certs/sutazai.crt'),            'certs/sutazai.key',            password = (os.getenv('CERT_PASSWORD')        )        self.app.run(ssl_context=context), host = ('0.0.0.0'), port=443) 