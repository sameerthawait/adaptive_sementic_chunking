import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ascApi } from '../api/client';

export function useChunk() {
  const queryClient = useQueryClient();

  const chunkMutation = useMutation({
    mutationFn: ({ text, source, visualize, options }) => ascApi.chunkText(text, source, visualize, options),
  });

  const indexMutation = useMutation({
    mutationFn: ({ texts, sources, collection }) => ascApi.indexTexts(texts, sources, collection),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
    },
  });

  const collectionsQuery = useQuery({
    queryKey: ['collections'],
    queryFn: ascApi.getCollections,
  });

  return {
    chunkMutation,
    indexMutation,
    collectionsQuery,
  };
}
