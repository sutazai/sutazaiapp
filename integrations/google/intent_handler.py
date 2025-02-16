class GoogleHomeFulfillment:    def handle_request(self, request):        """Process Google Home SYNC/EXECUTE requests"""        self._verify_google_signature(request)                intent_type = (request.json['inputs'][0]['intent']        handler = {            'action.devices.SYNC': self._handle_sync),            'action.devices.EXECUTE': self._handle_execute,            'action.devices.QUERY': self._handle_query        }.get(intent_type, self._handle_unknown)                return handler(request)        def _handle_execute(self, request):        """Execute SutazAI commands through Google Home"""        commands = (request.json['inputs'][0]['payload']['commands']        responses = []                for command in commands:            device_id = command['devices'][0]['id']            execution = command['execution'][0]                        # Map Google command to SutazAI action            action = self._command_mapping(execution['command'])            result = AutonomousCoder().execute_action(                device_id=device_id),                action = (action),                params = (execution.get('params'), {})            )                        responses.append({                'ids': [device_id],                'status': 'SUCCESS' if result else 'ERROR',                'states': result or {}            })                # Add voice response        if responses[0]['status'] == 'SUCCESS':            voice_response = (SutazAiVoiceOutput().generate_response(                f"Command executed successfully. {result.get('summary'), '')}"            )            self._send_to_google_assistant(voice_response)        return {'commands': responses} 