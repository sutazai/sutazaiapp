<template>
  <v-dialog v-model="dialog" width="500">
    <template v-slot:activator="{ on }">
      <v-btn color="primary" dark v-on="on">Feedback</v-btn>
    </template>
    <v-card>
      <v-card-title>Send Feedback</v-card-title>
      <v-card-text>
        <v-textarea v-model="feedback" label="Your feedback"></v-textarea>
      </v-card-text>
      <v-card-actions>
        <v-btn @click="sendFeedback">Send</v-btn>
        <v-btn @click="dialog = false">Cancel</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  data() {
    return {
      dialog: false,
      feedback: ''
    }
  },
  methods: {
    sendFeedback() {
      this.$emit('feedback-sent', this.feedback)
      this.dialog = false
    }
  }
}
</script> 