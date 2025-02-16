const aiInterface = {
  data() {
    return {
      searchMode: 'internal',
      searchQuery: '',
      results: [],
      chatHistory: [],
      currentExpression: 'neutral'
    }
  },
  methods: {
    async executeSearch() {
      this.isThinking = true
      this.currentExpression = 'processing'
      
      const results = await SutazAiSearchAPI.query({
        query: this.searchQuery,
        mode: this.searchMode,
        context: {
          user: FOUNDER.id,
          location: this.$geo.get(),
          currentProjects: this.$projects.active()
        }
      })
      
      this.results = results.map(result => ({
        ...result,
        aiAnalysis: this.analyzeResult(result)
      }))
      
      this.isThinking = false
      this.currentExpression = 'ready'
    },
    handleFileUpload(files) {
      files.forEach(file => {
        const preview = URL.createObjectURL(file)
        this.$store.commit('addAsset', {
          type: file.type.startsWith('image/') ? 'image' : 'document',
          preview,
          meta: this.$ai.analyzeFile(file)
        })
      })
    },
    async voiceCommand() {
      const command = await this.$voiceRecognition.start()
      this.searchQuery = command
      this.executeSearch()
    }
  }
} 