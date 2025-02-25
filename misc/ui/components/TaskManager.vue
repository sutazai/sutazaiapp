<template>
  <v-card>
    <v-card-title>Task Manager</v-card-title>
    <v-card-text>
      <v-list>
        <v-list-item v-for="task in tasks" :key="task.id">
          <v-list-item-content>
            <v-list-item-title>{{ task.title }}</v-list-item-title>
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
      tasks: []
    }
  },
  mounted() {
    this.fetchTasks()
  },
  methods: {
    fetchTasks() {
      // Fetch tasks from API
    }
  }
}
</script> 