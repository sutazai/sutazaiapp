<template>
  <v-card>
    <v-card-title>Real-Time Notifications</v-card-title>
    <v-card-text>
      <v-list>
        <v-list-item v-for="notification in notifications" :key="notification.id">
          <v-list-item-content>
            <v-list-item-title>{{ notification.message }}</v-list-item-title>
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
      notifications: []
    }
  },
  mounted() {
    this.fetchNotifications()
  },
  methods: {
    fetchNotifications() {
      // Fetch notifications from API
    }
  }
}
</script> 