<template>
  <div class="sutazai-dashboard">
    <div class="system-health">
      <h3>🛡️ SutazAI System Vital Signs</h3>
      <div class="metrics-grid">
        <metric-card title="CPU Usage" :value="metrics.cpu" type="percentage"/>
        <metric-card title="Security Status" :value="metrics.security" type="status"/>
        <metric-card title="Active Agents" :value="metrics.agents" type="count"/>
        <metric-card title="SutazAi Entanglement" :value="metrics.entanglement" type="stability"/>
      </div>
    </div>

    <div class="alert-feed">
      <h3>🚨 Active Alerts</h3>
      <alert-item v-for="alert in alerts" 
                  :alert="alert"
                  @acknowledge="handleAck"/>
    </div>

    <div class="security-overlay" v-if="securityView">
      <security-radar :threats="threats"/>
      <approval-requests :pending="pendingApprovals"/>
    </div>

    <div class="founder-status">
      <div class="verification-badge" :class="verificationStatus">
        {{ verificationStatus.toUpperCase() }}
      </div>
    </div>
  </div>
</template> 