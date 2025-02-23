<template>
  <v-card>
    <v-card-title>User Profile</v-card-title>
    <v-card-text>
      <v-form>
        <v-text-field v-model="user.name" label="Name"></v-text-field>
        <v-text-field v-model="user.email" label="Email"></v-text-field>
        <v-btn @click="saveProfile">Save</v-btn>
      </v-form>
    </v-card-text>
  </v-card>
</template>

<script>
export default {
  data() {
    return {
      user: {
        name: '',
        email: ''
      }
    }
  },
  methods: {
    saveProfile() {
      this.$emit('profile-saved', this.user)
    }
  }
}
</script> 