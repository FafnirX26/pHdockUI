"use client";

import { useState, useEffect, useRef } from "react";
import { Search, Loader2, Check, ChevronDown } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

interface ReceptorSearchResult {
  pdb_id: string;
  title: string;
  resolution: string;
  method: string;
  description: string;
}

interface ReceptorSearchProps {
  value: string;
  onChange: (pdbId: string) => void;
  placeholder?: string;
}

export default function ReceptorSearch({
  value,
  onChange,
  placeholder = "Search PDB database..."
}: ReceptorSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [selectedReceptor, setSelectedReceptor] = useState<ReceptorSearchResult | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounced search query
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Search receptors
  const { data: receptors, isLoading } = useQuery<ReceptorSearchResult[]>({
    queryKey: ["receptor-search", debouncedQuery],
    queryFn: async () => {
      if (!debouncedQuery || debouncedQuery.length < 2) {
        return [];
      }
      const response = await axios.get(`/api/receptors/search?query=${encodeURIComponent(debouncedQuery)}`);
      return response.data;
    },
    enabled: debouncedQuery.length >= 2
  });

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (receptor: ReceptorSearchResult) => {
    setSelectedReceptor(receptor);
    setSearchQuery(receptor.title);
    onChange(receptor.pdb_id);
    setIsOpen(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setSearchQuery(newValue);
    setIsOpen(true);

    // Clear selection if user modifies the input
    if (selectedReceptor && newValue !== selectedReceptor.title) {
      setSelectedReceptor(null);
    }
  };

  const handleInputFocus = () => {
    setIsOpen(true);
  };

  return (
    <div ref={containerRef} className="relative">
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          placeholder={placeholder}
          className="w-full px-3 py-2 pl-10 pr-10 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <Search
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
          size={18}
        />
        {isLoading && (
          <Loader2
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 animate-spin"
            size={18}
          />
        )}
        {selectedReceptor && !isLoading && (
          <Check
            className="absolute right-3 top-1/2 -translate-y-1/2 text-green-500"
            size={18}
          />
        )}
        {!selectedReceptor && !isLoading && searchQuery && (
          <ChevronDown
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400"
            size={18}
          />
        )}
      </div>

      {/* Dropdown Results */}
      {isOpen && searchQuery.length >= 2 && (
        <div className="absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 text-center text-gray-500 dark:text-gray-400">
              <Loader2 className="animate-spin mx-auto mb-2" size={20} />
              <p className="text-sm">Searching PDB database...</p>
            </div>
          ) : receptors && receptors.length > 0 ? (
            <div className="py-1">
              {receptors.map((receptor) => (
                <button
                  key={receptor.pdb_id}
                  onClick={() => handleSelect(receptor)}
                  className="w-full px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 last:border-b-0 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">
                          {receptor.pdb_id}
                        </span>
                        <span className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">
                          {receptor.resolution}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 dark:text-gray-300 truncate">
                        {receptor.title}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {receptor.method}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="p-4 text-center text-gray-500 dark:text-gray-400">
              <p className="text-sm">No receptors found</p>
              <p className="text-xs mt-1">Try searching by PDB ID, protein name, or organism</p>
            </div>
          )}
        </div>
      )}

      {/* Help text */}
      {!searchQuery && !isOpen && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Search by PDB ID (e.g., "1ATP"), protein name (e.g., "COX-2"), or organism
        </p>
      )}

      {/* Selected receptor info */}
      {selectedReceptor && (
        <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-xs">
          <div className="flex items-center gap-2">
            <Check size={14} className="text-green-600 dark:text-green-400" />
            <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">
              {selectedReceptor.pdb_id}
            </span>
            <span className="text-gray-600 dark:text-gray-400">•</span>
            <span className="text-gray-600 dark:text-gray-400">
              {selectedReceptor.resolution}
            </span>
            <span className="text-gray-600 dark:text-gray-400">•</span>
            <span className="text-gray-600 dark:text-gray-400">
              {selectedReceptor.method}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
