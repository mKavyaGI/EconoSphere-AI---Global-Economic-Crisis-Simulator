"use client"
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { useRouter } from 'next/navigation'
import L from 'leaflet'
import { Country } from '@/lib/api/countries'

// Fix for default marker icon in Leaflet + Next.js
const defaultIcon = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
})

interface MapProps {
  countries: Country[]
}

export default function WorldMap({ countries }: MapProps) {
  const router = useRouter()
  
  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-800 shadow-sm z-0 relative">
      <MapContainer center={[20, 0]} zoom={2} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {countries.map((c) => (
           c.metadata_info?.latitude && c.metadata_info?.longitude ? (
            <Marker 
              key={c.iso3} 
              position={[c.metadata_info.latitude, c.metadata_info.longitude]}
              icon={defaultIcon}
              eventHandlers={{
                click: () => {
                  router.push(`/dashboard/countries/${c.iso3}`)
                },
              }}
            >
              <Popup>
                <div className="font-semibold">{c.name}</div>
                <div className="text-sm text-gray-500">View Profile</div>
              </Popup>
            </Marker>
          ) : null
        ))}
      </MapContainer>
    </div>
  )
}
