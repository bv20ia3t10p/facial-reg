import { Badge } from 'antd';
import { ReactNode } from 'react';

interface NotificationBadgeProps {
  count: number;
  children: ReactNode;
}

export function NotificationBadge({ count, children }: NotificationBadgeProps) {
  return (
    <Badge 
      count={count} 
      offset={[-8, 8]}
      size="small"
      style={{ 
        backgroundColor: count > 0 ? '#f5222d' : 'transparent',
      }}
    >
      {children}
    </Badge>
  );
}
