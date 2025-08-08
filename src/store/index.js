/**
 * Central Store Index - Zustand State Management
 * 
 * This file exports all store modules and provides a unified interface
 * for accessing application state across components.
 */

export { useVoiceStore } from './voiceStore';
export { useTextInputStore } from './textInputStore';
export { useStreamingStore } from './streamingStore';
export { useConversationStore } from './conversationStore';
export { useSidebarStore } from './sidebarStore';

// Store types for TypeScript support
export type {
  VoiceState,
  VoiceActions,
  TextInputState, 
  TextInputActions,
  StreamingState,
  StreamingActions,
  ConversationState,
  ConversationActions,
  SidebarState,
  SidebarActions
} from './types';