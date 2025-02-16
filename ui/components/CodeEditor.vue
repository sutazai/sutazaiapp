<template>
  <div class="sutazai-editor">
    <div class="language-selector">
      <select v-model="currentLanguage">
        <option v-for="lang in languages" :value="lang">
          {{ lang.toUpperCase() }}
        </option>
      </select>
    </div>
    
    <div class="editor-container">
      <monaco-editor 
        v-model="code"
        :language="currentLanguage"
        :security-level="securityLevel"
        @analyze="realTimeAnalysis"
      />
    </div>

    <div class="execution-controls">
      <button @click="runCode">
        <i class="fas fa-play"></i> Run
      </button>
      <button @click="requestApproval" v-if="needsApproval">
        <i class="fas fa-shield-alt"></i> Get Approval
      </button>
    </div>
  </div>
</template> 