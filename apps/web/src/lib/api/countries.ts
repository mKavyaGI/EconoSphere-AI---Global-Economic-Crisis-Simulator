import { apiClient } from './client';

export interface CountryMetadata {
  capital: string | null;
  population: number | null;
  currency: string | null;
  flag_url: string | null;
  latitude?: number | null;
  longitude?: number | null;
}

export interface Country {
  iso3: string;
  name: string;
  region: string | null;
  income_group: string | null;
  metadata_info?: CountryMetadata | null;
}

export interface HistoricalIndicator {
  indicator_name: string;
  year: number;
  value: number;
  source: string | null;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

export const fetchCountries = async (offset = 0, limit = 50, region?: string): Promise<PaginatedResponse<Country>> => {
  const { data } = await apiClient.get('/countries', { params: { offset, limit, region } });
  return data;
};

export const fetchCountry = async (iso3: string): Promise<Country> => {
  const { data } = await apiClient.get(`/countries/${iso3}`);
  return data;
};

export const fetchCountryHistory = async (iso3: string, indicator?: string): Promise<HistoricalIndicator[]> => {
  const { data } = await apiClient.get(`/countries/${iso3}/history`, { params: { indicator } });
  return data;
};

export const searchCountries = async (query: string): Promise<Country[]> => {
  const { data } = await apiClient.get('/countries/search', { params: { query } });
  return data;
};
