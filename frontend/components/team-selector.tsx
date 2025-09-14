"use client"

import { useState } from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"
import type { TeamSelectorProps } from "@/types"

export function TeamSelector({ selectedTeam, onTeamChange, teams, label }: TeamSelectorProps) {
  const [open, setOpen] = useState(false)

  const selectedTeamData = teams.find((team) => team.id === selectedTeam)

  const groupedTeams = teams.reduce(
    (acc, team) => {
      const key = `${team.conference} ${team.division}`
      if (!acc[key]) {
        acc[key] = []
      }
      acc[key].push(team)
      return acc
    },
    {} as Record<string, typeof teams>,
  )

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">{label}</label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between h-auto p-3 bg-transparent"
          >
            {selectedTeamData ? (
              <div className="flex items-center space-x-3">
                <div
                  className="w-4 h-4 rounded-full border"
                  style={{ backgroundColor: selectedTeamData.primaryColor }}
                />
                <div className="text-left">
                  <div className="font-medium">
                    {selectedTeamData.city} {selectedTeamData.name}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {selectedTeamData.conference} {selectedTeamData.division}
                  </div>
                </div>
              </div>
            ) : (
              "Select team..."
            )}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0" align="start">
          <Command>
            <CommandInput placeholder="Search teams..." />
            <CommandList>
              <CommandEmpty>No team found.</CommandEmpty>
              {Object.entries(groupedTeams).map(([division, divisionTeams]) => (
                <CommandGroup key={division} heading={division}>
                  {divisionTeams.map((team) => (
                    <CommandItem
                      key={team.id}
                      value={`${team.city} ${team.name} ${team.abbreviation}`}
                      onSelect={() => {
                        onTeamChange(team.id)
                        setOpen(false)
                      }}
                      className="flex items-center space-x-3 p-3"
                    >
                      <div className="flex items-center space-x-3 flex-1">
                        <div className="w-4 h-4 rounded-full border" style={{ backgroundColor: team.primaryColor }} />
                        <div>
                          <div className="font-medium">
                            {team.city} {team.name}
                          </div>
                          <div className="text-xs text-muted-foreground">{team.abbreviation}</div>
                        </div>
                      </div>
                      <Check className={cn("h-4 w-4", selectedTeam === team.id ? "opacity-100" : "opacity-0")} />
                    </CommandItem>
                  ))}
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedTeamData && (
        <div className="flex items-center space-x-2 pt-2">
          <Badge variant="secondary" className="text-xs">
            {selectedTeamData.conference}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {selectedTeamData.division}
          </Badge>
        </div>
      )}
    </div>
  )
}
