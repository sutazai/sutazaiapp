<template>
  <div class="voice-controls">
    <button @click="toggleVoice" :class="{active: voiceActive}">
      <i class="fas fa-microphone"></i>
      {{ voiceActive ? 'Listening...' : 'Start Voice' }}
    </button>
    
    <div class="voice-feedback">
      <div v-if="isProcessing" class="processing-indicator">
        <i class="fas fa-brain"></i> Analyzing...
      </div>
      <div v-if="transcript" class="transcript">
        "{{ transcript }}"
      </div>
    </div>
  </div>
</template> 