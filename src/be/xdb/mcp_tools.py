"""
XDB MCP Tools - MCP wrapper for the Experience API (XAPI).

Provides tool definitions and routing for BE-facing XDB operations.
All tool calls are logged to the CAT audit log when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

from .audit_log import AuditLog
from .models import BudStatus, EventType, TagType
from .xdb import XDB

# Optional ASK access control
try:
    from src.ask.permissions import Actor, ActorType, Permission
    ASK_AVAILABLE = True
except ImportError:  # pragma: no cover - ASK not required for core usage
    ASK_AVAILABLE = False
    Actor = None
    ActorType = None
    Permission = None

logger = logging.getLogger(__name__)


@dataclass
class ToolAccessDecision:
    allowed: bool
    reason: str = ""
    permission: Optional[str] = None


class XDBMCPTools:
    """
    MCP tool definitions for XDB (Experience API).

    This is BE-facing only. CAT/auditor access is explicitly denied.
    """

    def __init__(
        self,
        xdb: XDB,
        *,
        audit_log: Optional[AuditLog] = None,
        actor: Optional[Any] = None,
    ):
        self.xdb = xdb
        self.audit_log = audit_log
        self.actor = actor

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions for XDB."""
        return [
            {
                "name": "xdb.record",
                "description": "Record a timestep (token, message, tool call, etc.) to the Experience Log.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "enum": ["input", "output", "tool_call", "tool_response", "steering", "system"],
                        },
                        "content": {"type": "string"},
                        "concept_activations": {"type": "object"},
                        "event_id": {"type": "string"},
                        "event_start": {"type": "boolean", "default": False},
                        "event_end": {"type": "boolean", "default": False},
                        "token_id": {"type": "integer"},
                        "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                    },
                    "required": ["event_type", "content"],
                },
            },
            {
                "name": "xdb.tag",
                "description": "Apply a folksonomy tag to a timestep, event, or tick range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tag_name_or_id": {"type": "string"},
                        "timestep_id": {"type": "string"},
                        "event_id": {"type": "string"},
                        "tick_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
                        "note": {"type": "string"},
                    },
                    "required": ["tag_name_or_id"],
                },
            },
            {
                "name": "xdb.create_tag",
                "description": "Create a new tag (concept, entity, bud, or custom).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tag_type": {"type": "string", "enum": ["concept", "entity", "bud", "custom"]},
                        "description": {"type": "string"},
                        "entity_type": {"type": "string", "enum": ["person", "organization", "place", "thing"]},
                        "related_concepts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "tag_type"],
                },
            },
            {
                "name": "xdb.comment",
                "description": "Add a comment to a timestep, event, or tick range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "timestep_id": {"type": "string"},
                        "event_id": {"type": "string"},
                        "tick_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "xdb.query",
                "description": "Query the Experience Log for timesteps matching filters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "tick_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "start_time": {"type": "string", "format": "date-time"},
                                "end_time": {"type": "string", "format": "date-time"},
                            },
                        },
                        "event_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["input", "output", "tool_call", "tool_response", "steering", "system"],
                            },
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "concept_activations": {"type": "object"},
                        "text_search": {"type": "string"},
                        "limit": {"type": "integer", "default": 100},
                    },
                },
            },
            {
                "name": "xdb.recent",
                "description": "Get the N most recent timesteps.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "default": 100},
                    },
                },
            },
            {
                "name": "xdb.tags",
                "description": "List tags, optionally filtered by type or status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tag_type": {"type": "string", "enum": ["concept", "entity", "bud", "custom"]},
                        "bud_status": {
                            "type": "string",
                            "enum": ["collecting", "ready", "training", "promoted", "abandoned"],
                        },
                    },
                },
            },
            {
                "name": "xdb.status",
                "description": "Get XDB state including tag counts, storage usage, and session info.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "xdb.concepts",
                "description": "List concepts from the concept pack, optionally filtered by parent.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "parent_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 100},
                    },
                },
            },
            {
                "name": "xdb.find_concept",
                "description": "Search for concepts by name.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "xdb.graph_neighborhood",
                "description": "Walk the concept graph from seed nodes, returning nodes and edges.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "seed_ids": {"type": "array", "items": {"type": "string"}},
                        "max_depth": {"type": "integer", "default": 2},
                        "direction": {"type": "string", "enum": ["both", "ancestors", "descendants"], "default": "both"},
                        "max_nodes": {"type": "integer", "default": 100},
                    },
                    "required": ["seed_ids"],
                },
            },
            {
                "name": "xdb.buds",
                "description": "List buds (candidate concepts) optionally filtered by status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["collecting", "ready", "training", "promoted", "abandoned"],
                        },
                    },
                },
            },
            {
                "name": "xdb.bud_examples",
                "description": "Get all timesteps tagged with a bud (training examples).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bud_tag_id": {"type": "string"},
                    },
                    "required": ["bud_tag_id"],
                },
            },
            {
                "name": "xdb.bud_ready",
                "description": "Mark a bud as ready for training (collecting -> ready).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bud_tag_id": {"type": "string"},
                    },
                    "required": ["bud_tag_id"],
                },
            },
            {
                "name": "xdb.pin",
                "description": "Pin timesteps to WARM storage for training data preservation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timestep_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["timestep_ids"],
                },
            },
            {
                "name": "xdb.unpin",
                "description": "Unpin timesteps from WARM storage.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timestep_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["timestep_ids"],
                },
            },
            {
                "name": "xdb.quota",
                "description": "Get WARM and COLD storage quota status.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "xdb.context",
                "description": "Get current context window state.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "xdb.compact",
                "description": "Manually trigger context window compaction.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "xdb.maintenance",
                "description": "Run storage maintenance (compression, cleanup).",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP tool call for XDB operations."""
        access = self._check_access(tool_name)
        if not access.allowed:
            result = {
                "error": "access_denied",
                "reason": access.reason,
                "permission": access.permission,
            }
            self._log_audit(tool_name, arguments, result, allowed=False)
            return result

        try:
            result = self._dispatch_tool(tool_name, arguments)
            self._log_audit(tool_name, arguments, result, allowed=True)
            return result
        except Exception as exc:
            error = {"error": "tool_failed", "detail": str(exc)}
            self._log_audit(tool_name, arguments, error, allowed=True)
            return error

    def _dispatch_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "xdb.record":
            event_type = EventType(arguments["event_type"])
            ts_id = self.xdb.record(
                event_type=event_type,
                content=arguments["content"],
                concept_activations=arguments.get("concept_activations"),
                event_id=arguments.get("event_id"),
                event_start=bool(arguments.get("event_start", False)),
                event_end=bool(arguments.get("event_end", False)),
                token_id=arguments.get("token_id"),
                role=arguments.get("role"),
            )
            return {"status": "recorded", "timestep_id": ts_id, "current_tick": self.xdb.current_tick}

        if tool_name == "xdb.tag":
            tick_range = self._parse_tick_range(arguments.get("tick_range"))
            app_id = self.xdb.tag(
                arguments["tag_name_or_id"],
                timestep_id=arguments.get("timestep_id"),
                event_id=arguments.get("event_id"),
                tick_range=tick_range,
                confidence=float(arguments.get("confidence", 1.0)),
                note=arguments.get("note"),
            )
            return {"status": "tagged", "application_id": app_id}

        if tool_name == "xdb.create_tag":
            tag_type = TagType(arguments["tag_type"])
            if tag_type == TagType.ENTITY:
                tag = self.xdb.create_entity_tag(
                    arguments["name"],
                    arguments.get("entity_type") or "unknown",
                    description=arguments.get("description"),
                )
            elif tag_type == TagType.BUD:
                tag = self.xdb.create_bud_tag(
                    arguments["name"],
                    arguments.get("description") or "",
                )
            else:
                tag = self.xdb.tag_index.create_tag(
                    arguments["name"],
                    tag_type,
                    description=arguments.get("description"),
                )
            return {"status": "created", "tag": tag.to_dict()}

        if tool_name == "xdb.comment":
            tick_range = self._parse_tick_range(arguments.get("tick_range"))
            comment_id = self.xdb.comment(
                arguments["content"],
                timestep_id=arguments.get("timestep_id"),
                event_id=arguments.get("event_id"),
                tick_range=tick_range,
            )
            return {"status": "commented", "comment_id": comment_id}

        if tool_name == "xdb.query":
            tick_range = self._parse_tick_range(arguments.get("tick_range"))
            time_range = self._parse_time_range(arguments.get("time_range"))
            event_types = None
            if arguments.get("event_types"):
                event_types = [EventType(et) for et in arguments["event_types"]]
            concept_filters = None
            if arguments.get("concept_activations"):
                concept_filters = {}
                for concept_id, bounds in arguments["concept_activations"].items():
                    if isinstance(bounds, dict):
                        concept_filters[concept_id] = (
                            float(bounds.get("min", 0.0)),
                            float(bounds.get("max", 1.0)),
                        )
                    else:
                        concept_filters[concept_id] = (0.5, 1.0)
            results = self.xdb.experience_log.query(
                xdb_id=self.xdb.xdb_id,
                tick_range=tick_range,
                time_range=time_range,
                event_types=event_types,
                tags=arguments.get("tags"),
                concept_activations=concept_filters,
                text_search=arguments.get("text_search"),
                limit=int(arguments.get("limit", 100)),
            )
            return {"count": len(results), "timesteps": [ts.to_dict() for ts in results]}

        if tool_name == "xdb.recent":
            results = self.xdb.recall_recent(int(arguments.get("n", 100)))
            return {"count": len(results), "timesteps": [ts.to_dict() for ts in results]}

        if tool_name == "xdb.tags":
            tag_type = TagType(arguments["tag_type"]) if arguments.get("tag_type") else None
            bud_status = BudStatus(arguments["bud_status"]) if arguments.get("bud_status") else None
            tags = self.xdb.tag_index.find_tags(tag_type=tag_type, bud_status=bud_status)
            return {"count": len(tags), "tags": [tag.to_dict() for tag in tags]}

        if tool_name == "xdb.status":
            return self.xdb.get_state()

        if tool_name == "xdb.concepts":
            parent_id = arguments.get("parent_id")
            results = self.xdb.browse_concepts(parent_id)
            limit = int(arguments.get("limit", 100))
            return {"count": len(results[:limit]), "concepts": [c.to_dict() for c in results[:limit]]}

        if tool_name == "xdb.find_concept":
            results = self.xdb.find_concept(arguments["query"])
            limit = int(arguments.get("limit", 20))
            return {"count": len(results[:limit]), "concepts": [c.to_dict() for c in results[:limit]]}

        if tool_name == "xdb.graph_neighborhood":
            result = self.xdb.tag_index.graph_neighborhood(
                seed_ids=arguments["seed_ids"],
                max_depth=int(arguments.get("max_depth", 2)),
                direction=arguments.get("direction", "both"),
                max_nodes=int(arguments.get("max_nodes", 100)),
            )
            return result

        if tool_name == "xdb.buds":
            status = BudStatus(arguments["status"]) if arguments.get("status") else None
            buds = self.xdb.get_buds(status=status)
            return {"count": len(buds), "buds": [b.to_dict() for b in buds]}

        if tool_name == "xdb.bud_examples":
            examples = self.xdb.get_bud_examples(arguments["bud_tag_id"])
            return {
                "bud_tag_id": arguments["bud_tag_id"],
                "count": len(examples),
                "examples": [e.to_dict() for e in examples],
            }

        if tool_name == "xdb.bud_ready":
            tag = self.xdb.mark_bud_ready(arguments["bud_tag_id"])
            if not tag:
                return {"error": "not_found", "bud_tag_id": arguments["bud_tag_id"]}
            return {"status": "ready", "tag": tag.to_dict()}

        if tool_name == "xdb.pin":
            pinned = self.xdb.pin_for_training(arguments["timestep_ids"], arguments.get("reason", ""))
            return {"status": "pinned", "pinned_count": pinned, "quota": self.xdb.get_warm_quota()}

        if tool_name == "xdb.unpin":
            unpinned = self.xdb.unpin_training_data(arguments["timestep_ids"])
            return {"status": "unpinned", "unpinned_count": unpinned, "quota": self.xdb.get_warm_quota()}

        if tool_name == "xdb.quota":
            return self.xdb.get_warm_quota()

        if tool_name == "xdb.context":
            return self.xdb.get_context_state()

        if tool_name == "xdb.compact":
            record = self.xdb.request_compaction()
            if record:
                return {"status": "compacted", "record": record.to_dict()}
            return {"status": "no_compaction_needed"}

        if tool_name == "xdb.maintenance":
            self.xdb.run_maintenance()
            return {"status": "maintenance_complete", "stats": self.xdb.storage_manager.get_stats()}

        raise ValueError(f"Unknown tool: {tool_name}")

    def _check_access(self, tool_name: str) -> ToolAccessDecision:
        if not ASK_AVAILABLE or not self.actor:
            return ToolAccessDecision(allowed=True)

        audit_only = {
            ActorType.CAT_SCALE,
            ActorType.EXTERNAL_OVERSIGHT,
            ActorType.AUTHORITY,
            ActorType.TRIBE,
        }
        if self.actor.actor_type in audit_only:
            return ToolAccessDecision(
                allowed=False,
                reason="actor_has_audit_only_access",
            )

        permission_map = {
            "read": Permission.OBSERVE_ENTRIES,
            "write": Permission.ADMIN_CONFIGURE,
            "compact": Permission.ADMIN_COMPACT,
        }

        read_tools = {
            "xdb.query",
            "xdb.recent",
            "xdb.tags",
            "xdb.status",
            "xdb.concepts",
            "xdb.find_concept",
            "xdb.graph_neighborhood",
            "xdb.buds",
            "xdb.bud_examples",
            "xdb.quota",
            "xdb.context",
        }
        write_tools = {
            "xdb.record",
            "xdb.tag",
            "xdb.create_tag",
            "xdb.comment",
            "xdb.bud_ready",
            "xdb.pin",
            "xdb.unpin",
        }
        compact_tools = {"xdb.compact", "xdb.maintenance"}

        if tool_name in read_tools:
            required = permission_map["read"]
        elif tool_name in write_tools:
            required = permission_map["write"]
        elif tool_name in compact_tools:
            required = permission_map["compact"]
        else:
            required = permission_map["read"]

        if not self.actor.has_permission(required):
            return ToolAccessDecision(
                allowed=False,
                reason="permission_denied",
                permission=required.name,
            )

        return ToolAccessDecision(allowed=True)

    def _log_audit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        *,
        allowed: bool,
    ) -> None:
        if not self.audit_log:
            return

        xdb_id = self.xdb.xdb_id or "xdb-default"
        tick = self.xdb.current_tick or 0
        actor_id = getattr(self.actor, "actor_id", "")
        payload = {
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "allowed": allowed,
            "actor_id": actor_id,
            "timestamp": datetime.now().isoformat(),
        }
        raw_content = json.dumps(payload, sort_keys=True)

        try:
            self.audit_log.record(
                xdb_id=xdb_id,
                tick=tick,
                event_type=EventType.TOOL_CALL if allowed else EventType.SYSTEM,
                raw_content=raw_content,
                lens_activations={},
                steering_applied=[],
            )
        except Exception as exc:
            logger.error(f"Failed to write audit log for {tool_name}: {exc}")

    @staticmethod
    def _parse_tick_range(tick_range: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        if not tick_range:
            return None
        if isinstance(tick_range, list) and len(tick_range) == 2:
            return int(tick_range[0]), int(tick_range[1])
        start = tick_range.get("start")
        end = tick_range.get("end")
        if start is None or end is None:
            return None
        return int(start), int(end)

    @staticmethod
    def _parse_time_range(time_range: Optional[Dict[str, Any]]) -> Optional[Tuple[datetime, datetime]]:
        if not time_range:
            return None
        start_time = time_range.get("start_time")
        end_time = time_range.get("end_time")
        if not start_time or not end_time:
            return None
        start_time = start_time.replace("Z", "+00:00")
        end_time = end_time.replace("Z", "+00:00")
        return datetime.fromisoformat(start_time), datetime.fromisoformat(end_time)
