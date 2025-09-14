"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Trophy, BarChart3 } from "lucide-react"
import { motion } from "framer-motion"

export function Navigation() {
  const pathname = usePathname()

  return (
    <motion.header
      className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50"
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="container mx-auto px-4 py-4">
        <nav className="flex items-center justify-between">
          <motion.div
            className="flex items-center space-x-2"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, repeatDelay: 5 }}
            >
              <Trophy className="h-8 w-8 text-primary" />
            </motion.div>
            <div>
              <h1 className="text-xl font-bold text-foreground">NFL Prediction Engine</h1>
              <p className="text-sm text-muted-foreground">2024-2025 Season</p>
            </div>
          </motion.div>

          <div className="flex items-center space-x-2">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button variant={pathname === "/" ? "default" : "ghost"} asChild className="flex items-center space-x-2">
                <Link href="/">
                  <Trophy className="h-4 w-4" />
                  <span>Predictions</span>
                </Link>
              </Button>
            </motion.div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant={pathname === "/performance" ? "default" : "ghost"}
                asChild
                className="flex items-center space-x-2"
              >
                <Link href="/performance">
                  <BarChart3 className="h-4 w-4" />
                  <span>Model Performance</span>
                </Link>
              </Button>
            </motion.div>
          </div>
        </nav>
      </div>
    </motion.header>
  )
}
