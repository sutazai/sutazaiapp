<template>
  <div v-if="hasError">
    <v-alert type="error">
      Something went wrong. Please try again later.
    </v-alert>
  </div>
  <slot v-else></slot>
</template>

<script>
export default {
  data() {
    return {
      hasError: false
    }
  },
  errorCaptured(err, vm, info) {
    this.hasError = true
    console.error('Error:', err, 'Info:', info)
    return false
  }
}
</script> 