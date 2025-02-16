<template>
  <v-card>
    <v-card-title>Collaboration</v-card-title>
    <v-card-text>
      <v-list>
        <v-list-item v-for="user in onlineUsers" :key="user.id">
          <v-list-item-content>
            <v-list-item-title>{{ user.name }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-card-text>
  </v-card>
</template>

<script>
export default {
  data() {
    return {
      onlineUsers: []
    }
  },
  mounted() {
    this.fetchOnlineUsers()
  },
  methods: {
    fetchOnlineUsers() {
      // Fetch online users from API
    }
  }
}
</script> 