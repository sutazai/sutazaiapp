<template>
  <div class="message" :class="[type, mood]">
    <div class="avatar">
      <img v-if="mood === 'happy'" src="smile-holo.png">
      <img v-if="mood === 'curious'" src="thinking-holo.png">
    </div>
    <div class="content">
      <p>{{ message }}</p>
      <div class="reaction-buttons">
        <button @click="sendLove">❤️</button>
        <button @click="requestHug">🤗</button>
      </div>
    </div>
  </div>
</template> 