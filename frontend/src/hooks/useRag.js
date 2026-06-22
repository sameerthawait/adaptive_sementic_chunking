import { useMutation } from '@tanstack/react-query';
import { ascApi } from '../api/client';

export function useRag() {
  const queryMutation = useMutation({
    mutationFn: ({ query, collection, k, useRag, ...options }) => {
      if (useRag) {
        return ascApi.queryRag(query, collection, k, true, options);
      } else {
        return ascApi.queryRetrieve(query, collection, k, options);
      }
    },
  });

  return {
    queryMutation,
  };
}
