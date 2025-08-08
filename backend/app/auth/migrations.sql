-- Add missing columns to users table for JWT authentication
-- Run this migration to update the existing users table

-- Add is_admin column if it doesn't exist
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;

-- Add last_login column if it doesn't exist
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS last_login TIMESTAMP WITH TIME ZONE;

-- Add failed_login_attempts column if it doesn't exist
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0;

-- Add locked_until column if it doesn't exist
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP WITH TIME ZONE;

-- Create index on is_admin for faster admin queries
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin) WHERE is_admin = true;

-- Update existing users to have admin flag (optional - set first user as admin)
-- UPDATE users SET is_admin = true WHERE id = 1;

-- Create a default admin user (password: Admin123!)
-- Note: The password hash is for 'Admin123!' using bcrypt
INSERT INTO users (username, email, password_hash, is_active, is_admin) 
VALUES (
    'admin', 
    'admin@sutazai.local',
    '$2b$12$YQZ8qKxV8g.xZFN7bQ5Kz.0JOJ3iZ5uQZKXzHWxxC.bRtEiNF0gDi',
    true,
    true
) ON CONFLICT (username) DO NOTHING;