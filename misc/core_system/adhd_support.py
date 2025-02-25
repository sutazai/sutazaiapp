class NeurodivergentSupport:
    def __init__(self):
        self.focus_assist = HyperfocusFacilitator()
        self.task_anchors = TemporalAnchorSystem()

    def adhd_priority_boost(self, task):
        """ADHD-friendly task management"""
        return {
            "task": task,
            "reminders": self._calculate_optimal_reminders(task),
            "motivation_elements": self._add_engagement_elements(task),
        }
