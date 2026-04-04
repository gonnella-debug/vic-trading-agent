# PROJECT STATUS — GG's AI Bot Fleet

> **Last updated:** 2026-04-04
> **Purpose:** Complete system reference so any new Claude chat can understand the entire operation without explanation from GG.

---

## TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Current Status of Every Component](#2-current-status-of-every-component)
3. [Recent Changes](#3-recent-changes)
4. [Pending Tasks](#4-pending-tasks)
5. [Architecture Diagram](#5-architecture-diagram)
6. [Key Decisions Made](#6-key-decisions-made)

---

## 1. SYSTEM OVERVIEW

### 1.1 All Bots & Services

| Bot/Service | Role | Repo | Language | Framework |
|-------------|------|------|----------|-----------|
| **Sara** | WhatsApp sales bot for **Nucassa Real Estate** (Dubai property buyer/seller qualification) | `SARA-BOT/bot.py` | Python | FastAPI |
| **Simon** | WhatsApp bot for **Nucassa Holdings Ltd** (institutional investor relations, ADGM SPVs, $1M+ minimum) | `SARA-BOT/simon_bot_v1.py` | Python | FastAPI |
| **Lester** | WhatsApp bot for **ListR.ae** (UAE property marketplace — buyers, sellers, agents) | `LESTER-BOT/bot.py` (latest) / `SARA-BOT/lester_bot_v1.py` (older) | Python | FastAPI |
| **Alex** | AI Chief of Staff — GG's single Telegram interface to manage everything. Health monitoring, content direction, lead analytics, briefings, blast orchestration | `manager-agent/alex_v2.py` | Python | FastAPI + APScheduler |
| **Mark** | AI Marketing Bot — content generation, image rendering (Pillow), social media posting (IG/FB/LinkedIn), Telegram approval workflow | `mark-bot/mark_bot_final.py` | Python | FastAPI + Pillow |
| **Vic** | BTC/USDT perpetual futures scalping agent on Hyperliquid. 4 strategies, 10x leverage, $500 account | `vic-trading-agent/vic.py` | Python | FastAPI + Hyperliquid SDK |
| **WhatsApp Scraper** | Watches local folder for real estate brochures, uses Claude to classify, uploads to Google Drive by developer/project | `whatsapp-scraper/scraper.py` | Python | Google Drive API |
| **DLD Valuator** | Property valuation API using Dubai Land Department transaction data (808 transactions) | `whatsapp-scraper/dld-valuator/main.py` | Python | FastAPI + Pandas |

### 1.2 Railway Deployment URLs

| Service | Railway URL |
|---------|------------|
| Sara (Nucassa RE) | `https://sarah-bot-production.up.railway.app` |
| Simon (Nucassa Holdings) | `https://web-production-4b93df.up.railway.app` |
| Lester (ListR.ae) | `https://web-production-e207a.up.railway.app` |
| Alex (Manager) | `https://alex-cos-manager-production.up.railway.app` |
| Mark (Marketing) | `https://mark-bot-production.up.railway.app` (+ Cloudflare tunnel for dev) |
| Vic (Trading) | `https://vic-trading-agent-production.up.railway.app` |
| DLD Valuator | `https://web-production-915cf.up.railway.app` |

### 1.3 GitHub Repos

| Repo | Contents | Key Files |
|------|----------|-----------|
| `gonnella-debug/SARA-BOT` | Sara, Simon, Lester (older version) | `bot.py` (Sara, 1129 lines), `simon_bot_v1.py` (1175 lines), `lester_bot_v1.py` (981 lines) |
| `gonnella-debug/manager-agent` | Alex (Chief of Staff) | `alex_v2.py` (1745 lines), `manager.py` (legacy), `test_bitrix_webhook.py` |
| `gonnella-debug/mark-bot` | Mark (Marketing) | `mark_bot_final.py` (2631 lines), `fonts/`, `tunnel.sh`, `start_mark.sh` |
| `gonnella-debug/vic-trading-agent` | Vic (Trading) | `vic.py` (3255 lines), `backtest.py`, `backtest_results.json` |
| `gonnella-debug/LESTER-BOT` | Lester (latest standalone version) | `bot.py` (1139 lines, has email auto-reply), `lester_bot_v1.py` (older) |
| `gonnella-debug/whatsapp-scraper` | Brochure scraper + DLD Valuator | `scraper.py`, `dld-valuator/main.py`, `dld_transactions.csv` |

### 1.4 Airtable Bases & IDs

| Base | Airtable Base ID | Tables | Used By |
|------|-----------------|--------|---------|
| Sara (Nucassa RE) | `appObd5iy4paEJ8xY` | `Conversations`, `BlastLog` | Sara bot, Alex |
| Simon (Holdings) | `appVBSWWg4UFLdIv6` | `Conversations`, `BlastLog` | Simon bot, Alex |
| Lester (ListR.ae) | `appp6OGY1Gqq8JPmD` | `Conversations`, `BlastLog` | Lester bot, Alex |

**Conversations table fields (Sara — most complete):**
Phone, History (JSON), Status (ACTIVE/QUALIFIED/NOT_INTERESTED/DNC/FUTURE_LEAD), ContactName, Budget, Purpose, PropertyType, Urgency, Language, Intent (BUYER/SELLER/B&S), Location, Building, Bedrooms, Size, AskingPrice, SellTimeLine, LastMessage, LastActivity

**Simon:** Phone, History, Status, ContactName, Language, LastMessage, LastActivity

**Lester:** Phone, History, Status, ContactName, Language, Intent, LastMessage, LastActivity

### 1.5 API Keys & Credentials (Names Only — No Values)

**Shared across bots:**
- `CLAUDE_API_KEY` — Anthropic Claude API (claude-sonnet-4-20250514 model)
- `META_ACCESS_TOKEN` — Meta/Facebook WhatsApp Business API
- `META_PHONE_NUMBER_ID` — WhatsApp phone number ID
- `META_VERIFY_TOKEN` — WhatsApp webhook verification
- `META_APP_SECRET` — Meta app secret
- `AIRTABLE_API_KEY` — Airtable REST API
- `BITRIX_WEBHOOK_URL` — Bitrix24 CRM webhook

**Sara/Simon/Lester specific:**
- `AIRTABLE_BASE_ID` — Per-bot Airtable base
- `SARA_EMAIL_APP_PASSWORD` — Gmail app password (IMAP polling)
- `SIMON_EMAIL_APP_PASSWORD` — Gmail app password (IMAP polling)
- `LESTER_EMAIL_APP_PASSWORD` — Gmail app password (IMAP polling)
- `RESEND_API_KEY` — Resend email sending API

**Alex specific:**
- `TELEGRAM_BOT_TOKEN` — Alex's Telegram bot
- `TELEGRAM_CHAT_ID` — GG's Telegram chat (`8511419437`)

**Mark specific:**
- `TELEGRAM_BOT_TOKEN` — Mark's Telegram bot (separate from Alex)
- `TELEGRAM_CHAT_ID` — GG's Telegram chat
- `META_SYSTEM_TOKEN` — Meta system token for IG/FB posting
- `LI_CLIENT_ID` / `LI_CLIENT_SECRET` / `LI_ACCESS_TOKEN` / `LI_REFRESH_TOKEN` — LinkedIn OAuth
- `GDRIVE_API_KEY` / `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` / `GDRIVE_REFRESH_TOKEN` — Google Drive
- `GOOGLE_SERVICE_ACCOUNT_JSON` — Drive service account
- `GDRIVE_FOLDER_ID` / `GDRIVE_LISTR_REF_ID` / `GDRIVE_NUCASSA_REF_ID` — Drive folder IDs
- `UNSPLASH_ACCESS_KEY` — Unsplash (fallback only)

**Vic specific:**
- `HL_WALLET_ADDRESS` — Hyperliquid wallet
- `HL_PRIVATE_KEY` — Hyperliquid private key
- `TELEGRAM_BOT_TOKEN` — Vic's Telegram bot (separate)
- `TELEGRAM_CHAT_ID` — GG's Telegram chat
- `CLAUDE_API_KEY` — For AI Market Brain
- `WEBHOOK_SECRET` — API endpoint authentication
- `TRADING_MODE` — "paper" or "live"
- `LEVERAGE` — Set to 10

**WhatsApp Scraper:**
- `CLAUDE_API_KEY` — Document classification
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — Notifications
- `oauth_credentials.json` / `token.json` — Google Drive OAuth

### 1.6 Telegram Bots & Purposes

| Telegram Bot | Purpose | Used By |
|-------------|---------|---------|
| Alex Bot | GG's primary management interface. Receives briefings, content suggestions, lead alerts, health alerts. GG sends commands and approves content. | `manager-agent/alex_v2.py` |
| Mark Bot | Content approval workflow. Sends rendered image previews with POST NOW / POST 6PM / Reject / Regenerate buttons. | `mark-bot/mark_bot_final.py` |
| Vic Bot | Trading notifications. Trade entry/exit alerts, daily summaries, `/journal`, `/metrics`, `/regime`, `/news`, `/intelligence`, `/closeall` commands. Claude-powered market Q&A. | `vic-trading-agent/vic.py` |
| Scraper Bot | Notifies GG when brochures are processed and uploaded to Drive. | `whatsapp-scraper/scraper.py` |

**All bots send to GG's Telegram chat ID: `8511419437`**

### 1.7 Google Sheets (Blast Contact Sources)

| Bot | Spreadsheet ID | Tabs |
|-----|---------------|------|
| Sara | `18XJ7f9JtdMT3ERhXTpCGeZnmPtSDJYFuy5ZfvXfyM9U` | Contacts1–4 |
| Simon | `1EkPJfR5kgLkh9XgwS5EMXlAzbHC9SawNiHB9dtOK9zI` | Contacts1–4 |
| Lester | `1WedbMucbi6XcI4sbd4VBw3EPqrGnnjhX4Em1b1bshjo` | Contacts1–4 |

### 1.8 Google Drive Folders

| Folder | ID | Purpose |
|--------|-----|---------|
| Sarah's Projects (root) | `1QoloKwEVPojBMfkTcSkbRL1ryo0a8jif` | Brochure uploads organized by developer/project |
| Nucassa Reference Images | env `GDRIVE_NUCASSA_REF_ID` | Mark bot reference photos |
| ListR Reference Images | env `GDRIVE_LISTR_REF_ID` | Mark bot reference photos |

### 1.9 Bitrix24 CRM

- **Webhook token:** `waj54ro8quxn4zkzbv3k9lh4ldjlbvca`
- **Alex inbound webhook:** POST `/bx7x9k2m` (receives ONCRMLEADADD, ONCRMLEADUPDATE)
- **Status codes:** QUALIFIED = `UC_ZJDAGQ`, NOT_INTERESTED = `UC_Y4IUXG`, WON, IN_PROCESS
- **All three bots** push qualified leads to Bitrix via `crm.lead.add`

### 1.10 Internal Team Numbers & Emails

| Person | Phone | Emails | Role |
|--------|-------|--------|------|
| GG | `971585286821` | `gonnella@nucassa.com`, `21gonnella@gmail.com` | Founder/Chairman — full access, bypasses qualification |
| Riff/Arif | `971585620980` | `arif@nucassa.com` | Team member — bypasses qualification |

### 1.11 Bot Email Addresses

| Bot | Gmail (IMAP) | Outbound (Resend) |
|-----|-------------|-------------------|
| Sara | `saranucassa@gmail.com` | `sara@nucassa.com` |
| Simon | `simonnucassa@gmail.com` | `simon@nucassa.holdings` |
| Lester | `lester.listr@gmail.com` | `lester@list.ae` |

---

## 2. CURRENT STATUS OF EVERY COMPONENT

### 2.1 LIVE & WORKING

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **Sara** | 6.0 | ✅ LIVE | WhatsApp + Email auto-reply + Daily blast (250/day) + Bitrix CRM |
| **Simon** | 2.1 | ✅ LIVE | WhatsApp + Email auto-reply + LinkedIn outreach endpoint + Daily blast (250/day) |
| **Lester** | 2.0 | ✅ LIVE | WhatsApp + DLD valuations + Daily blast (250/day in SARA-BOT, 100/day in LESTER-BOT) |
| **Alex** | 2.0 | ✅ LIVE | Telegram interface, health monitoring (5min), briefings (9am/6pm), blast verification (3pm), intelligence scan (3hr), content pipeline to Mark |
| **Mark** | Latest | ✅ LIVE | Pillow rendering, 3 brands, IG/FB/LinkedIn posting, Telegram approval, Google Drive upload |
| **Vic** | 3.0 | ✅ LIVE | 4 strategies on Hyperliquid, 10x leverage, $500 account, AI Market Brain, Telegram commands |
| **DLD Valuator** | 1.0 | ✅ LIVE | 808 transactions, area/property_type/bedrooms valuation |
| **WhatsApp Scraper** | 1.0 | ✅ LIVE | Local service watching `~/Desktop/WhatsApp Brochures`, Claude classification, Drive upload |
| **Bitrix24 Integration** | — | ✅ LIVE | All bots push leads, Alex receives webhook events |
| **Airtable** | — | ✅ LIVE | 3 bases, conversation history + blast logs |

### 2.2 PARTIALLY BUILT / LIMITATIONS

| Component | Issue |
|-----------|-------|
| **DLD Valuator** | Only 808 transactions — needs Dubai Pulse API key to upgrade to 1.5M transactions |
| **Simon LinkedIn** | `/linkedin-outreach` endpoint exists but needs dedicated LinkedIn account (currently uses GG's) |
| **PhantomBuster** | Free tier — scraped 75 UHNW profiles but no phone numbers. Needs paid tier |
| **Mark state** | In-memory only — posting log and render state lost on redeploy |
| **Mark PDF post** | `list_drive_pdfs()` function called but never defined — PDF post endpoint broken |
| **LESTER-BOT vs SARA-BOT** | Two copies of Lester with different blast limits (100 vs 250) and templates. Need to sync |

### 2.3 KNOWN ISSUES

| Issue | Severity | Component |
|-------|----------|-----------|
| Email processed_ids set grows unbounded (capped at 5000 manually) | Low | Sara/Simon/Lester |
| Airtable History field capped at 90,000 chars — oldest messages truncated | Low | All bots |
| No Bitrix error recovery — if lead creation fails, conversation still marked QUALIFIED | Medium | All bots |
| Telegram markdown parsing — retries without markdown if first send fails | Low | Alex, Mark, Vic |
| No GDPR data deletion mechanism | Medium | All bots |
| Vic Telegram polling restarts max 5 times then gives up | Medium | Vic |
| Vic in-memory strategy state lost on restart (journal persists) | Low | Vic |
| No CI/CD pipeline for any repo | Low | All |

---

## 3. RECENT CHANGES (Last 7 Days — March 28 to April 4, 2026)

### SARA-BOT (Sara, Simon, Lester)

| Date | Change |
|------|--------|
| Apr 2 | **Blast daily limit increased from 100 to 250** for all bots |
| Apr 2 | **WhatsApp templates updated** to v34 (sara34, simon34, lester34) |
| Apr 2 | HTTP retry helper with exponential backoff on 429/5xx |
| Apr 1 | **Email auto-reply system** — IMAP polling every 60s → Claude → Resend response |
| Apr 1 | **LinkedIn outreach endpoint** added for Simon (PhantomBuster contacts) |
| Apr 1 | Internal email recognition for GG and Arif (bypass qualification) |
| Apr 1 | Switched from SMTP to Resend HTTP API for all email sending |
| Apr 1 | Email signature cleanup — removed broken image logos |

### Manager-Agent (Alex)

| Date | Change |
|------|--------|
| Apr 3 | **Full operational awareness** — Mark posting log, Drive activity, Vic status, message volumes (parallel fetch) |
| Apr 3 | **3pm blast verification** — alerts GG if any bot missed its blast |
| Apr 2 | New template awareness (sara34/simon34/lester34 performance tracking) |
| Apr 2 | **Alex must decide, not ask** — picks one angle per brand, sends approve/reject buttons |
| Apr 2 | Separate content suggestions per brand (one Telegram message each) |
| Apr 2 | Retry logic for Mark calls, Mark-down alerts on failure |
| Apr 2 | **Autonomous intelligence system** — 6 web searches every 3 hours (DLD, geopolitics, mortgages, HNI migration, competitors, sentiment) |
| Apr 2 | Web search capability added to Alex |

### Mark-Bot

| Date | Change |
|------|--------|
| Apr 3-4 | Instagram 500 error retry with backoff |
| Apr 3-4 | Instagram error 9007 (media not ready) retry |
| Apr 3 | **Real curated photos** — 74 verified Unsplash URLs, killed API search |
| Apr 3 | Holdings-specific and ListR-specific photo categories |
| Apr 2-3 | Full-bleed photo + gradient rendering on slides 1 & 2 |
| Apr 2 | Randomised layout variations for visual variety |
| Apr 2 | Increased line spacing for readability |

### Vic Trading Agent

| Date | Change |
|------|--------|
| Apr 3-4 | **SL/TP overhaul**: 2% SL ($10), 4% TP1 partial ($20), 6% TP2 full ($30) |
| Apr 3 | Leverage read from env var (set to 10 on Railway) |
| Apr 3 | Leverage shown on `/status` endpoint |
| Apr 2-3 | Multiple Docker rebuild fixes for Railway |
| Apr 2 | Block opposite-side trades, `/closeall` manual override |
| Apr 2 | Fix phantom position loop — validates Hyperliquid state before close |

### Naming Changes

- **Simon → Emma**: Simon is referred to as "Emma" externally to clients. Code still uses "Simon" internally. No code rename has been done yet.

---

## 4. PENDING TASKS (Priority Order)

| # | Task | Priority | Component |
|---|------|----------|-----------|
| 1 | **Dubai Pulse API key** — upgrade DLD Valuator from 808 to 1.5M transactions | HIGH | DLD Valuator |
| 2 | **Sync LESTER-BOT and SARA-BOT** — reconcile blast limits (100 vs 250) and templates | HIGH | Lester |
| 3 | **Fix Mark PDF post** — `list_drive_pdfs()` undefined, breaks `/pdf_post` endpoint | HIGH | Mark |
| 4 | **Simon → Emma rename** — update external-facing name in system prompts while keeping code name | MEDIUM | Simon |
| 5 | **PhantomBuster upgrade** — paid tier to extract phone numbers from LinkedIn profiles | MEDIUM | Simon |
| 6 | **Dedicated LinkedIn account for Simon/Emma** — separate from GG's personal account | MEDIUM | Simon |
| 7 | **Persistent state for Mark** — posting log survives redeploy (database or file) | MEDIUM | Mark |
| 8 | **Persistent state for Vic** — strategy metrics survive restart | MEDIUM | Vic |
| 9 | **GDPR data deletion endpoint** — ability to purge contact data from Airtable | LOW | All bots |
| 10 | **CI/CD pipelines** — GitHub Actions for automated deploy + tests | LOW | All repos |
| 11 | **Bitrix lead creation error recovery** — retry or flag failed leads | LOW | All bots |

---

## 5. ARCHITECTURE DIAGRAM

### 5.1 Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GG (FOUNDER)                                    │
│                                                                              │
│  Telegram ◄──────────────────────────────────────────────────────────────┐   │
│  (Primary Interface)                                                     │   │
└──────┬───────────────────┬───────────────────┬───────────────────────────┘   │
       │                   │                   │                               │
       ▼                   ▼                   ▼                               │
┌─────────────┐   ┌──────────────┐   ┌──────────────┐                        │
│    ALEX     │   │     MARK     │   │     VIC      │                        │
│ Chief of    │──▶│  Marketing   │   │   Trading    │                        │
│ Staff       │   │  Bot         │   │   Agent      │                        │
│             │   │              │   │              │                        │
│ • Briefings │   │ • Rendering  │   │ • 4 Strats   │                        │
│ • Health    │   │ • IG/FB/LI   │   │ • Hyperliquid│                        │
│ • Content   │   │ • Approval   │   │ • AI Brain   │                        │
│ • Intel     │   │ • Drive save │   │ • PnL track  │                        │
│ • Alerts    │   │              │   │              │                        │
└──┬──┬──┬────┘   └──────────────┘   └──────────────┘                        │
   │  │  │                                                                     │
   │  │  └─────────────────────────────────────────────────────────────────────┘
   │  │
   │  │  ┌─────────────────────────────────────────────────────────────┐
   │  │  │                    WHATSAPP BOTS                            │
   │  │  │                                                             │
   │  ▼  ▼                                                             │
   │  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
   │  │   SARA   │    │  SIMON   │    │  LESTER  │                   │
   │  │  v6.0    │    │  v2.1    │    │  v2.0    │                   │
   │  │          │    │(Emma ext)│    │          │                   │
   │  │ Nucassa  │    │ Nucassa  │    │ ListR.ae │                   │
   │  │ Real Est │    │ Holdings │    │ Market   │                   │
   │  │          │    │          │    │ place    │                   │
   │  │ Buyers   │    │ Investors│    │ Buyers   │                   │
   │  │ Sellers  │    │ $1M+ SPV │    │ Sellers  │                   │
   │  │          │    │          │    │ Agents   │                   │
   │  └────┬─────┘    └────┬─────┘    └────┬─────┘                   │
   │       │               │               │                          │
   │       └───────────────┼───────────────┘                          │
   │                       │                                          │
   │                       ▼                                          │
   │  ┌──────────────────────────────────────────────────────────┐   │
   │  │                  SHARED SERVICES                          │   │
   │  │                                                          │   │
   │  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
   │  │  │Airtable │  │ Bitrix24 │  │  Resend  │  │  Gmail  │ │   │
   │  │  │3 bases  │  │   CRM    │  │  Email   │  │  IMAP   │ │   │
   │  │  └─────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
   │  │                                                          │   │
   │  │  ┌─────────┐  ┌──────────┐  ┌──────────┐              │   │
   │  │  │  Meta   │  │  Google  │  │   DLD    │              │   │
   │  │  │WhatsApp │  │  Sheets  │  │Valuator  │              │   │
   │  │  │  API    │  │(contacts)│  │  (808tx) │              │   │
   │  │  └─────────┘  └──────────┘  └──────────┘              │   │
   │  └──────────────────────────────────────────────────────────┘   │
   │                                                                  │
   │  ┌──────────────────────────────────────────────────────────┐   │
   │  │               WHATSAPP SCRAPER                            │   │
   │  │  ~/Desktop/WhatsApp Brochures → Claude classify →        │   │
   │  │  Google Drive (Sarah's Projects folder)                   │   │
   │  └──────────────────────────────────────────────────────────┘   │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

### 5.2 Lead Flow: WhatsApp → CRM

```
Customer sends WhatsApp message
        │
        ▼
Meta WhatsApp Cloud API (webhook)
        │
        ▼
Bot (Sara/Simon/Lester) receives message
        │
        ├──▶ Fetch conversation history from Airtable
        │
        ├──▶ Build Claude prompt with system instructions + history
        │
        ├──▶ Claude generates response (+ qualification JSON if ready)
        │
        ├──▶ Send response back via WhatsApp API
        │
        ├──▶ Save updated history to Airtable
        │
        └──▶ If QUALIFIED → Push lead to Bitrix24 CRM
                    │
                    ▼
              Alex receives Bitrix webhook (ONCRMLEADADD)
                    │
                    ▼
              Alex alerts GG via Telegram
```

### 5.3 Alex → Mark Content Pipeline

```
Alex generates content suggestions (based on web search + market intel)
        │
        ▼
Alex sends 3 Telegram messages (one per brand) with Approve/Reject buttons
        │
        ▼
GG taps "Approve" on a suggestion
        │
        ▼
Alex sends POST /generate to Mark:
  { brand: "nucassa_re", content_type: "carousel", topic: "..." }
        │
        ▼
Mark generates content via Claude API
        │
        ▼
Mark renders slides with Pillow (1080x1350, full-bleed photo + gradient)
        │
        ▼
Mark sends preview images to GG via Telegram
  [POST NOW] [POST 6PM] [Reject] [Regenerate]
        │
        ▼
GG taps "POST NOW" or "POST 6PM"
        │
        ▼
Mark posts to Instagram + Facebook + LinkedIn (per brand config)
        │
        ▼
Mark saves approved posts to Google Drive
        │
        ▼
Alex queries Mark /posting_log for evening briefing
```

### 5.4 Emma/Simon LinkedIn Flow

```
PhantomBuster scrapes LinkedIn profiles (institutional investors)
        │
        ▼
Contact data extracted: name, title, company, email, LinkedIn URL
        │
        ▼
POST /linkedin-outreach to Simon bot with contacts array
        │
        ▼
Simon generates personalized email per contact via Claude
  (references their title, company, introduces Nucassa Holdings SPVs)
        │
        ▼
Email sent via Resend API from simon@nucassa.holdings
        │
        ▼
If investor replies → Gmail IMAP poller picks up → Claude auto-reply
        │
        ▼
Conversation continues via email until qualified → Bitrix CRM
```

### 5.5 Alex's Scheduled Operations

| Time (Dubai) | Action |
|-------------|--------|
| 1:00 AM | Scan Google Drive for new brochures, recommend content angles |
| 9:00 AM | **Morning briefing** — overnight leads, news, content suggestions, focus |
| 10:00 AM (UTC) | Daily blast trigger for Sara, Simon, Lester (WhatsApp templates) |
| Every 5 min | Health check all bots (ping endpoints) |
| Every 10 min | Alert on new qualified leads |
| Every 30 min | Cross-brand opportunities (Sara $1M+ → Holdings pitch), uncontacted leads |
| Every 3 hours | Intelligence scan (6 web searches: DLD, geopolitics, mortgages, HNI, competitors, sentiment) |
| 3:00 PM | **Blast verification** — alert if any bot missed its daily blast |
| 6:00 PM | **Evening briefing** — pipeline check, Mark posting log, Vic PnL, Drive activity, ops report |

### 5.6 Vic Trading Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    VIC TRADING AGENT                            │
│                                                                │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌───────────┐ │
│  │ TV Scanner  │ │RSI Diverge  │ │BB Squeeze│ │VWAP Bounce│ │
│  │ (5m + 1h)   │ │ (5m)        │ │ (5m)     │ │ (1m)      │ │
│  └──────┬──────┘ └──────┬──────┘ └────┬─────┘ └─────┬─────┘ │
│         │               │              │              │        │
│         └───────────────┼──────────────┼──────────────┘        │
│                         │              │                        │
│                         ▼              ▼                        │
│                 ┌──────────────────────────┐                   │
│                 │    REGIME FILTER          │                   │
│                 │ TRENDING/RANGING/         │                   │
│                 │ TRANSITIONAL/VOLATILE     │                   │
│                 └────────────┬─────────────┘                   │
│                              │                                  │
│                              ▼                                  │
│                 ┌──────────────────────────┐                   │
│                 │   AI MARKET BRAIN        │                   │
│                 │ Claude + web_search tool  │                   │
│                 │ (pre-trade risk check)    │                   │
│                 └────────────┬─────────────┘                   │
│                              │                                  │
│                              ▼                                  │
│                 ┌──────────────────────────┐                   │
│                 │   HYPERLIQUID            │                   │
│                 │ BTC/USDT Perpetual       │                   │
│                 │ 10x leverage             │                   │
│                 │ SL/TP on exchange        │                   │
│                 └──────────────────────────┘                   │
│                                                                │
│  Risk: 2% SL ($10) | 4% TP1 50% ($20) | 6% TP2 full ($30)   │
│  Max daily loss: 10% ($50) | Max trades/day: 4                │
│  Account: $500 | Leverage: 10x                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. KEY DECISIONS MADE

### 6.1 Technology Choices

| Decision | Why |
|----------|-----|
| **FastAPI for all bots** | Async-native, lightweight, handles webhooks efficiently on Railway's free tier |
| **Airtable over Postgres** | Visual interface for GG to inspect conversations directly. Easy API. Good enough for current scale |
| **Resend over SMTP** | SMTP (port 465/587) was unreliable from Railway. Resend HTTP API is more reliable and handles deliverability |
| **Pillow over Playwright/Canva** | Playwright was heavy and slow for Railway. Canva API was explored but Pillow gives full control and renders in <2s |
| **Hyperliquid over OKX** | Vic migrated from OKX. Hyperliquid has native SDK, better for perpetuals, transparent orderbook |
| **Claude Sonnet over GPT** | Better at following complex system prompts, qualification logic, multilingual (Arabic/English) |
| **Claude Haiku for Alex** | Faster responses for operational queries, cheaper for high-frequency health checks |
| **Telegram over WhatsApp for management** | WhatsApp is for customers. Telegram gives inline buttons, no message limits, better for operational workflows |
| **Google Sheets for blast contacts** | GG can manually edit contact lists. CSV export via public share URL — no auth needed |
| **APScheduler in Alex** | Built-in to Python, survives Railway restarts, handles Dubai timezone scheduling |

### 6.2 Architectural Decisions

| Decision | Why |
|----------|-----|
| **One repo per bot (mostly)** | Sara/Simon/Lester share a repo because they share WhatsApp infra. Alex, Mark, Vic are independent |
| **Alex as single orchestrator** | GG talks to one bot (Alex), not six. Alex decides what to tell Mark, when to check bots, etc. |
| **No database for Mark** | Content is ephemeral — render, approve, post, save to Drive. In-memory state is acceptable (for now) |
| **Exchange-level SL/TP for Vic** | Trigger orders placed on Hyperliquid itself — if Vic crashes, positions are still protected |
| **74 curated Unsplash URLs** | Unsplash API search returned irrelevant images. Curated list ensures every photo is Dubai/luxury/corporate appropriate |
| **Separate Telegram message per brand** | Forces GG to consciously approve/reject each brand's content individually |

### 6.3 What Was Tried & Rejected

| Approach | Why Rejected |
|----------|-------------|
| **Playwright for Mark rendering** | Too heavy for Railway, slow cold starts, unreliable headless browser on containerized environments |
| **Canva API for content** | Explored but didn't provide enough programmatic control over layouts |
| **Unsplash API search at runtime** | Returned low-quality, irrelevant images (food, tourists, animals). Replaced with curated verified URLs |
| **SMTP for email sending** | Ports 465/587 blocked or unreliable from Railway containers. Switched to Resend HTTP API |
| **OKX for Vic trading** | Migrated to Hyperliquid — better perpetuals API, native SDK, transparent orderbook |
| **Single combined briefing** | Alex used to send one big morning message. Split into structured sections for readability |
| **Alex presenting multiple content options** | GG didn't want to choose. Alex now picks one angle per brand and presents approve/reject only |
| **Logos in email signatures** | Images rendered broken in many email clients. Removed — text-only signatures now |
| **Multiple small PRs for refactors** | GG prefers one bundled PR for related changes — less churn |

---

## APPENDIX: Quick Reference

### All Endpoints by Service

**Sara (SARA-BOT/bot.py):**
- `GET /` — Health check (v6.0)
- `GET /webhook` — WhatsApp verification
- `POST /webhook` — Incoming WhatsApp messages
- `POST /send-email` — Send single email
- `POST /email-blast` — Bulk email campaign
- `POST /outreach` — WhatsApp template outreach
- `POST /blast` — Daily contact blast

**Simon (SARA-BOT/simon_bot_v1.py):**
- `GET /` — Health check (v2.1)
- `GET /webhook` — WhatsApp verification
- `POST /webhook` — Incoming WhatsApp messages
- `POST /send-email` — Send single email
- `POST /email-blast` — Bulk email campaign
- `POST /linkedin-outreach` — LinkedIn cold emails via PhantomBuster contacts
- `POST /blast` — Daily contact blast

**Lester (LESTER-BOT/bot.py or SARA-BOT/lester_bot_v1.py):**
- `GET /` — Health check (v2.0)
- `GET /webhook` — WhatsApp verification
- `POST /webhook` — Incoming WhatsApp messages (+ DLD valuation detection)
- `POST /send-email` — Send single email
- `POST /email-blast` — Bulk email campaign
- `POST /blast` — Daily contact blast

**Alex (manager-agent/alex_v2.py):**
- `GET /` — Health check (v2.0)
- `POST /bx7x9k2m` — Bitrix24 CRM webhook (rate limited: 10/60s/IP)

**Mark (mark-bot/mark_bot_final.py):**
- `GET /` — Health check + LinkedIn status + pending approvals
- `GET /status` — Batch status for all brands
- `GET /posting_log` — Today's posting history
- `GET /img/{image_id}` — Serve temp images for IG uploads
- `GET /linkedin/auth` — LinkedIn OAuth initiation
- `GET /linkedin/callback` — LinkedIn OAuth callback
- `POST /generate` — Generate content (called by Alex)
- `POST /approve_batch` — Approve & push batch
- `POST /pdf_post` — Generate from PDF brochure

**Vic (vic-trading-agent/vic.py):**
- `GET /` — Health check
- `GET /status` — Full trading status (strategies, metrics, positions)
- `GET /journal` — Trade journal history
- `POST /go_live` — Switch paper → live (requires WEBHOOK_SECRET)
- `POST /pause` — Pause trading (requires WEBHOOK_SECRET)
- `POST /resume` — Resume trading (requires WEBHOOK_SECRET)
- `POST /close_all` — Emergency close positions (requires WEBHOOK_SECRET)
- `POST /test_trade` — Place minimal test SHORT

**DLD Valuator (whatsapp-scraper/dld-valuator/main.py):**
- `GET /` — Health check + transaction count
- `GET /areas` — List all Dubai areas
- `GET /valuate` — Property valuation (area, property_type, bedrooms)
- `GET /widget` — Alias for /valuate

### Brand Social Media Config (Mark)

| Brand | IG Handle | Platforms | Color | Accent | Posting Days |
|-------|-----------|-----------|-------|--------|-------------|
| Nucassa RE | @nucassadubai | IG, FB, LI | #1C1C1C | #CDA17F (rose gold) | Mon/Wed/Fri 6pm Dubai |
| Nucassa Holdings | @nucassaholdings_ltd | IG, FB, LI | #1C1C1C | #CDA17F | Mon/Wed/Fri 6pm Dubai |
| ListR.ae | @listr.ae | IG, FB | #000000 | #B8962E (gold) | Tue/Thu/Sat 6pm Dubai |

### LinkedIn Company Page IDs (Mark)

- Nucassa Real Estate: `90919312`
- Nucassa Holdings: `109941216`

### WhatsApp Template Names (Current)

- Sara: `sara34`
- Simon: `simon34`
- Lester: `lester34` (SARA-BOT) / `lester_agent_outreach` (LESTER-BOT)
