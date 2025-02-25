<template>
  <div class="min-h-screen bg-background text-text">
    <NavBar />
    <RouterView />
    <Notifications />
  </div>
</template>

<script setup>
import { RouterView } from 'vue-router'
import NavBar from './components/NavBar.vue'
import Notifications from './components/Notifications.vue'
</script>

<style>
@import './assets/theme.css';

body {
  @apply bg-background text-text;
  font-family: 'Inter', sans-serif;
}

.router-link-active {
  @apply text-primary font-medium;
}

.router-link-exact-active {
  @apply text-primary font-bold;
}
</style> 