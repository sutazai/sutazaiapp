<template>
  <v-card>
    <v-card-title>Real-Time Data Visualization</v-card-title>
    <v-card-text>
      <line-chart :data="chartData" :options="chartOptions"></line-chart>
    </v-card-text>
  </v-card>
</template>

<script>
import { Line } from 'vue-chartjs'

export default {
  extends: Line,
  data() {
    return {
      chartData: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        datasets: [
          {
            label: 'Data',
            backgroundColor: '#f87979',
            data: [40, 39, 10, 40, 39]
          }
        ]
      },
      chartOptions: {
        responsive: true,
        maintainAspectRatio: false
      }
    }
  }
}
</script> 