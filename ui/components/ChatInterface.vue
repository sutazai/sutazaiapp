<template>
  <div class="sutazai-chat">
    <div class="chat-history">
      <message-bubble 
        v-for="msg in messages"
        :message="msg"
        :auth-status="authStatus"
        @approve="handleApproval"
      />
    </div>
    
    <div class="input-area">
      <auth-status-indicator :status="authStatus" />
      <chat-input 
        @send="processInput"
        :disabled="awaitingApproval"
      />
      <mfa-selector v-if="showAuthOptions" @select="initiateAuth" />
    </div>
  </div>
</template> 