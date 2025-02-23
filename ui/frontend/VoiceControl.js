export default {
  data() {
    return {
      isListening: false,
      lastCommand: '',
      processing: false
    }
  },
  methods: {
    toggleVoice() {
      this.isListening = !this.isListening
      this.$voiceEngine[this.isListening ? 'start' : 'stop']()
    },
    
    handleVoiceCommand(text) {
      this.processing = true
      this.lastCommand = text
      
      this.$nextTick(() => {
        this.$commandBus.send(text)
        this.processing = false
      })
    }
  }
} 