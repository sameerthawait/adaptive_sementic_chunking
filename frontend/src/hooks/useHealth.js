import { useQuery } from '@tanstack/react-query';
import { ascApi } from '../api/client';

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ascApi.getHealth,
    refetchInterval: 30000, // check health every 30s
  });
}
