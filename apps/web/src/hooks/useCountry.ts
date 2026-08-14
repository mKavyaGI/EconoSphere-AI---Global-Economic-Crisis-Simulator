import { useQuery } from '@tanstack/react-query';
import { fetchCountry, fetchCountryHistory } from '@/lib/api/countries';

export const useCountry = (iso3: string) => {
  return useQuery({
    queryKey: ['country', iso3],
    queryFn: () => fetchCountry(iso3),
  });
};

export const useCountryHistory = (iso3: string, indicator?: string) => {
  return useQuery({
    queryKey: ['country', iso3, 'history', indicator],
    queryFn: () => fetchCountryHistory(iso3, indicator),
  });
};
