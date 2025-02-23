class VoiceControlMixin {
  methods: {
    initVoice() {
      this.voiceRecognition = new WebSpeechRecognition({
        continuous: true,
        interimResults: true,
        commands: {
          'search *term': (term) => this.searchQuery = term,
          'open document': this.uploadDoc,
          'analyze this': this.analyzeSelection,
          'enable security': () => this.$security.enable(),
          'call sutazai': this.summonAI
        }
      })
      
      this.voiceRecognition.on('error', this.handleVoiceError)
    },
    
    toggleVoice() {
      this.voiceActive = !this.voiceActive
      this.voiceRecognition[this.voiceActive ? 'start' : 'stop']()
    }
  }
} 