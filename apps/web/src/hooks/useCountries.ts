import { useQuery } from '@tanstack/react-query';
import { fetchCountries, searchCountries } from '@/lib/api/countries';

export const useCountries = (offset = 0, limit = 50, region?: string) => {
  return useQuery({
    queryKey: ['countries', { offset, limit, region }],
    queryFn: () => fetchCountries(offset, limit, region),
  });
};

export const useSearchCountries = (query: string) => {
  return useQuery({
    queryKey: ['countries', 'search', query],
    queryFn: () => searchCountries(query),
    enabled: query.length >= 2,
  });
};
