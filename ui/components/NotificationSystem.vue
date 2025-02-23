<template>
  <div class="notification-system">
    <!-- Notification Settings -->
    <div class="notification-settings">
      <h3>Notification Settings</h3>
      <div class="setting">
        <label>
          <input type="checkbox" v-model="settings.enabled" />
          Enable Notifications
        </label>
      </div>
      <div class="setting">
        <label>
          Notification Frequency:
          <select v-model="settings.frequency">
            <option value="instant">Instant</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
          </select>
        </label>
      </div>
      <div class="setting">
        <label>
          Notification Types:
          <select v-model="settings.types" multiple>
            <option value="system">System Alerts</option>
            <option value="updates">Updates</option>
            <option value="errors">Errors</option>
            <option value="success">Success</option>
          </select>
        </label>
      </div>
      <button @click="saveSettings">Save Settings</button>
    </div>

    <!-- Notification Display -->
    <div class="notification-display">
      <h3>Notifications</h3>
      <div v-for="(notification, index) in notifications" :key="index" class="notification" :class="notification.type">
        <span>{{ notification.message }}</span>
        <button @click="dismissNotification(index)">Dismiss</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      settings: {
        enabled: true,
        frequency: 'instant',
        types: ['system', 'updates', 'errors', 'success']
      },
      notifications: []
    };
  },
  methods: {
    saveSettings() {
      localStorage.setItem('notificationSettings', JSON.stringify(this.settings));
      this.$emit('settings-updated', this.settings);
      this.showNotification('Notification settings saved successfully.', 'success');
    },
    showNotification(message, type) {
      if (this.settings.enabled && this.settings.types.includes(type)) {
        this.notifications.push({ message, type, timestamp: new Date() });
      }
    },
    dismissNotification(index) {
      this.notifications.splice(index, 1);
    }
  },
  mounted() {
    const savedSettings = localStorage.getItem('notificationSettings');
    if (savedSettings) {
      this.settings = JSON.parse(savedSettings);
    }
  }
};
</script>

<style scoped>
.notification-system {
  max-width: 400px;
  margin: 20px auto;
  padding: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
}

.notification-settings {
  margin-bottom: 20px;
}

.setting {
  margin-bottom: 10px;
}

.notification-display {
  margin-top: 20px;
}

.notification {
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.notification.system {
  background-color: #e3f2fd;
}

.notification.updates {
  background-color: #fff3e0;
}

.notification.errors {
  background-color: #ffebee;
}

.notification.success {
  background-color: #e8f5e9;
}

button {
  background-color: #007bff;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #0056b3;
}
</style> 