#!/usr/bin/env python3
"""
AAC Reddit API Documentation
============================

Comprehensive documentation for Reddit API endpoints used in the AAC arbitrage system.
This file serves as the central repository for Reddit API specifications and usage.

Reddit API Base: https://www.reddit.com/dev/api/
Rate Limits:
- Authenticated users: 600 requests per 10 minutes
- Non-authenticated users: 60 requests per hour

OAuth2 Scopes Used:
- read: Access to read Reddit content
- identity: Access to user identity information
- submit: Ability to post/submit content
- privatemessages: Access to private messages
"""

# Reddit API Endpoints Documentation
REDDIT_API_ENDPOINTS = {
    "GET_hot": {
        "endpoint": "GET [/r/subreddit]/hot",
        "oauth_scope": "read",
        "rss_support": True,
        "description": """
        Returns a listing of hot posts from the specified subreddit.
        Hot posts are determined by reddit's ranking algorithm which considers
        recency, score (upvotes - downvotes), and comment activity.
        """,
        "parameters": {
            "g": {
                "type": "string",
                "description": "Geographic region filter",
                "values": [
                    "GLOBAL", "US", "AR", "AU", "BG", "CA", "CL", "CO", "HR", "CZ", "FI", "FR", "DE", "GR", "HU", "IS",
                    "IN", "IE", "IT", "JP", "MY", "MX", "NZ", "PH", "PL", "PT", "PR", "RO", "RS", "SG", "ES", "SE", "TW",
                    "TH", "TR", "GB", "US_WA", "US_DE", "US_DC", "US_WI", "US_WV", "US_HI", "US_FL", "US_WY",
                    "US_NH", "US_NJ", "US_NM", "US_TX", "US_LA", "US_NC", "US_ND", "US_NE", "US_TN",
                    "US_NY", "US_PA", "US_CA", "US_NV", "US_VA", "US_CO", "US_AK", "US_AL", "US_AR",
                    "US_CT", "US_DE", "US_FL", "US_GA", "US_HI", "US_ID", "US_IL", "US_IN", "US_IA",
                    "US_KS", "US_KY", "US_LA", "US_MA", "US_MD", "US_ME", "US_MI", "US_MN", "US_MO",
                    "US_MS", "US_MT", "US_NC", "US_ND", "US_NE", "US_NH", "US_NJ", "US_NM", "US_NV",
                    "US_NY", "US_OH", "US_OK", "US_OR", "US_PA", "US_RI", "US_SC", "US_SD", "US_TN",
                    "US_TX", "US_UT", "US_VA", "US_VT", "US_WA", "US_WI", "US_WV", "US_WY"
                ],
                "required": False
            },
            "after": {
                "type": "string",
                "description": "Fullname of a thing - used for pagination",
                "required": False
            },
            "before": {
                "type": "string",
                "description": "Fullname of a thing - used for pagination",
                "required": False
            },
            "count": {
                "type": "integer",
                "description": "Number of items already seen in this listing",
                "default": 0,
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of items to return",
                "default": 25,
                "maximum": 100,
                "required": False
            },
            "show": {
                "type": "string",
                "description": "Optional parameter; if 'all' is passed, filters such as 'hide links that I have voted on' will be disabled",
                "values": ["all"],
                "required": False
            },
            "sr_detail": {
                "type": "boolean",
                "description": "Expand subreddits in the response",
                "required": False
            }
        },
        "response_format": {
            "kind": "Listing",
            "data": {
                "after": "string (fullname for pagination)",
                "before": "string (fullname for pagination)",
                "children": "array of post objects",
                "dist": "integer (number of posts returned)"
            }
        },
        "aac_usage": """
        Used in PRAWRedditClient.get_hot_posts() method to retrieve trending posts
        from WallStreetBets for sentiment analysis and arbitrage signal generation.
        Hot posts represent the most engaging and timely discussions that may
        indicate market sentiment shifts.
        """,
        "implementation_notes": """
        - PRAW library abstracts the raw API calls
        - Default limit is 100 posts for AAC sentiment analysis
        - Posts are processed for ticker extraction and sentiment scoring
        - Rate limiting is handled automatically by PRAW
        """
    },

    "GET_new": {
        "endpoint": "GET [/r/subreddit]/new",
        "oauth_scope": "read",
        "rss_support": True,
        "description": "Returns a listing of new posts from the specified subreddit, sorted by creation time.",
        "parameters": {
            "after": {"type": "string", "description": "Fullname for pagination"},
            "before": {"type": "string", "description": "Fullname for pagination"},
            "count": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 25, "maximum": 100},
            "show": {"type": "string", "values": ["all"]},
            "sr_detail": {"type": "boolean"}
        },
        "aac_usage": "Potential use for real-time sentiment monitoring of latest posts"
    },

    "GET_rising": {
        "endpoint": "GET [/r/subreddit]/rising",
        "oauth_scope": "read",
        "rss_support": True,
        "description": "Returns a listing of rising posts from the specified subreddit.",
        "parameters": {
            "after": {"type": "string", "description": "Fullname for pagination"},
            "before": {"type": "string", "description": "Fullname for pagination"},
            "count": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 25, "maximum": 100},
            "show": {"type": "string", "values": ["all"]},
            "sr_detail": {"type": "boolean"}
        },
        "aac_usage": "Potential use for identifying emerging sentiment trends"
    },

    "GET_top": {
        "endpoint": "GET [/r/subreddit]/top",
        "oauth_scope": "read",
        "rss_support": True,
        "description": "Returns a listing of top posts from the specified subreddit.",
        "parameters": {
            "t": {
                "type": "string",
                "values": ["hour", "day", "week", "month", "year", "all"],
                "description": "Time period for top posts"
            },
            "after": {"type": "string", "description": "Fullname for pagination"},
            "before": {"type": "string", "description": "Fullname for pagination"},
            "count": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 25, "maximum": 100},
            "show": {"type": "string", "values": ["all"]},
            "sr_detail": {"type": "boolean"}
        },
        "aac_usage": "Potential use for historical sentiment analysis"
    },

    "GET_comments": {
        "endpoint": "GET [/r/subreddit]/comments/article",
        "oauth_scope": "read",
        "rss_support": True,
        "description": "Get the comment tree for a given Link article.",
        "parameters": {
            "article": {"type": "string", "description": "ID36 of a link", "required": True},
            "comment": {"type": "string", "description": "ID36 of a comment (optional)"},
            "context": {"type": "integer", "description": "Number of parent comments", "min": 0, "max": 8},
            "depth": {"type": "integer", "description": "Maximum depth of subtrees"},
            "limit": {"type": "integer", "description": "Maximum number of comments"},
            "showedits": {"type": "boolean"},
            "showmedia": {"type": "boolean"},
            "showmore": {"type": "boolean"},
            "showtitle": {"type": "boolean"},
            "sort": {"type": "string", "values": ["confidence", "top", "new", "controversial", "old", "random", "qa", "live"]},
            "sr_detail": {"type": "boolean"},
            "theme": {"type": "string", "values": ["default", "dark"]},
            "threaded": {"type": "boolean"},
            "truncate": {"type": "integer", "min": 0, "max": 50}
        },
        "aac_usage": "Used in PRAWRedditClient.get_post_comments() for detailed sentiment analysis"
    },

    "GET_about": {
        "endpoint": "GET /r/{subreddit}/about",
        "oauth_scope": "read",
        "rss_support": False,
        "description": """
        Returns information about the specified subreddit.
        This includes subscriber count, description, rules, moderation settings,
        and other subreddit metadata.
        """,
        "parameters": {},
        "response_format": {
            "kind": "t5 (subreddit object)",
            "data": {
                "id": "string (base36 ID without t5_ prefix)",
                "name": "string (fullname, e.g., t5_2th52)",
                "display_name": "string (subreddit name)",
                "display_name_prefixed": "string (e.g., r/wallstreetbets)",
                "title": "string (subreddit title)",
                "public_description": "string (sidebar text)",
                "description": "string (detailed description)",
                "subscribers": "integer (subscriber count)",
                "created": "float (Unix timestamp)",
                "created_utc": "float (Unix timestamp)",
                "over18": "boolean (NSFW flag)",
                "restrict_posting": "boolean (posting restrictions)",
                "free_form_reports": "boolean (free-form reporting enabled)",
                "wiki_enabled": "boolean (wiki enabled)",
                "community_reviewed": "boolean (community reviewed)",
                "primary_color": "string (hex color)",
                "key_color": "string (hex color)",
                "icon_img": "string (icon URL)",
                "community_icon": "string (community icon URL)",
                "banner_img": "string (banner URL)",
                "submit_text": "string (submission guidelines)",
                "submit_text_html": "string (HTML submission guidelines)",
                "subreddit_type": "string (public/private/restricted)",
                "suggested_comment_sort": "string (default sort)",
                "url": "string (subreddit URL path)",
                "lang": "string (language code)",
                "user_is_subscriber": "boolean or null",
                "user_is_moderator": "boolean or null",
                "user_is_contributor": "boolean or null",
                "user_is_banned": "boolean or null",
                "user_is_muted": "boolean or null"
            }
        },
        "aac_usage": """
        Used to gather subreddit metadata for sentiment analysis context.
        Provides subscriber counts, community guidelines, and moderation settings
        that help interpret the sentiment and credibility of posts.
        """,
        "implementation_notes": """
        - No parameters required - just the subreddit name in the URL path
        - Response includes extensive subreddit configuration data
        - Useful for understanding community size, rules, and posting guidelines
        - Can help assess the credibility and context of sentiment data
        """
    }
}

# Reddit API Response Structures
REDDIT_RESPONSE_STRUCTURES = {
    "post_object": {
        "id": "string (base36 ID)",
        "name": "string (fullname, e.g., t3_abc123)",
        "title": "string",
        "selftext": "string (post content)",
        "url": "string (link URL)",
        "score": "integer (upvotes - downvotes)",
        "num_comments": "integer",
        "created_utc": "float (Unix timestamp)",
        "author": "string (username) or null",
        "subreddit": "string (subreddit name)",
        "subreddit_name_prefixed": "string (e.g., r/wallstreetbets)",
        "is_self": "boolean (true for text posts)",
        "over_18": "boolean (NSFW flag)",
        "spoiler": "boolean",
        "stickied": "boolean",
        "locked": "boolean",
        "archived": "boolean"
    },

    "comment_object": {
        "id": "string (base36 ID)",
        "name": "string (fullname, e.g., t1_abc123)",
        "body": "string (comment text)",
        "score": "integer",
        "created_utc": "float (Unix timestamp)",
        "author": "string (username) or null",
        "parent_id": "string (fullname of parent)",
        "replies": "object (nested comments)",
        "depth": "integer",
        "controversiality": "integer (0 or 1)",
        "is_submitter": "boolean"
    },

    "listing_response": {
        "kind": "Listing",
        "data": {
            "after": "string or null",
            "before": "string or null",
            "children": "array of post/comment objects",
            "dist": "integer (number of items returned)",
            "modhash": "string (for CSRF protection)"
        }
    },

    "subreddit_object": {
        "kind": "t5",
        "data": {
            "id": "string (base36 ID without t5_ prefix, e.g., '2th52')",
            "name": "string (fullname with t5_ prefix, e.g., 't5_2th52')",
            "display_name": "string (subreddit name, e.g., 'wallstreetbets')",
            "display_name_prefixed": "string (prefixed name, e.g., 'r/wallstreetbets')",
            "title": "string (subreddit title)",
            "public_description": "string (sidebar/public description)",
            "description": "string (detailed description with markdown)",
            "description_html": "string (HTML version of description)",
            "public_description_html": "string (HTML version of public description)",
            "subscribers": "integer (current subscriber count)",
            "created": "float (creation timestamp)",
            "created_utc": "float (UTC creation timestamp)",
            "over18": "boolean (NSFW/adult content flag)",
            "restrict_posting": "boolean (posting restrictions enabled)",
            "free_form_reports": "boolean (free-form reporting allowed)",
            "wiki_enabled": "boolean (wiki functionality enabled)",
            "community_reviewed": "boolean (community reviewed status)",
            "original_content_tag_enabled": "boolean (original content tagging)",
            "primary_color": "string (hex color code for theme)",
            "key_color": "string (accent hex color code)",
            "icon_img": "string (icon image URL)",
            "community_icon": "string (community icon URL)",
            "banner_img": "string (banner image URL)",
            "banner_background_image": "string (banner background URL)",
            "mobile_banner_image": "string (mobile banner URL)",
            "submit_text": "string (submission guidelines in markdown)",
            "submit_text_html": "string (HTML submission guidelines)",
            "submit_text_label": "string (submit text button label)",
            "submit_link_label": "string (submit link button label)",
            "subreddit_type": "string ('public', 'private', 'restricted', 'gold_restricted', 'archived', 'employees_only', 'gold_only', 'user')",
            "suggested_comment_sort": "string (default comment sort: 'confidence', 'top', 'new', 'controversial', 'old', 'random', 'qa', 'live')",
            "url": "string (subreddit URL path)",
            "lang": "string (language code, e.g., 'en')",
            "user_is_subscriber": "boolean or null (current user's subscription status)",
            "user_is_moderator": "boolean or null (current user's moderator status)",
            "user_is_contributor": "boolean or null (current user's contributor status)",
            "user_is_banned": "boolean or null (current user's ban status)",
            "user_is_muted": "boolean or null (current user's mute status)",
            "user_can_flair_in_sr": "boolean or null (user can set flair)",
            "user_flair_enabled_in_sr": "boolean (user flair enabled)",
            "link_flair_enabled": "boolean (link flair enabled)",
            "allow_images": "boolean (image posts allowed)",
            "allow_videos": "boolean (video posts allowed)",
            "allow_galleries": "boolean (gallery posts allowed)",
            "allow_polls": "boolean (poll posts allowed)",
            "show_media": "boolean (media preview enabled)",
            "show_media_preview": "boolean (media preview in feed)",
            "spoilers_enabled": "boolean (spoiler tags enabled)",
            "emojis_enabled": "boolean (custom emojis enabled)",
            "collapse_deleted_comments": "boolean (deleted comments collapsed)",
            "comment_score_hide_mins": "integer (minutes to hide comment scores)",
            "should_archive_posts": "boolean (posts auto-archived)",
            "notification_level": "string or null (notification preference)",
            "accept_followers": "boolean (follower requests accepted)",
            "allow_discovery": "boolean (discoverable in searches)",
            "advertiser_category": "string (advertising category)",
            "hide_ads": "boolean (ads hidden)",
            "prediction_leaderboard_entry_type": "integer (prediction settings)",
            "videostream_links_count": "integer (video stream links)",
            "is_crosspostable_subreddit": "boolean (crossposting allowed)"
        }
    }
}

# AAC-Specific Reddit API Usage
AAC_REDDIT_USAGE = {
    "primary_endpoint": "GET_hot",
    "primary_subreddit": "wallstreetbets",
    "subreddit_metadata_usage": {
        "subscriber_count": "19.7M+ subscribers - indicates market influence scale",
        "community_guidelines": "Strict rules against manipulation, spam, and low-effort content",
        "flair_system": "Comprehensive flair system for post categorization (DD, YOLO, Discussion, etc.)",
        "sentiment_context": "High engagement community with strong contrarian views",
        "market_timing": "Peak activity during market hours, especially pre-market and after-hours",
        "content_quality": "High signal-to-noise ratio due to community moderation",
        "risk_indicators": "Flair types indicate risk levels (YOLO = high risk, DD = analytical)",
        "engagement_metrics": "High comment counts indicate strong community interest"
    },
    "data_collection": {
        "post_frequency": "Every 5-15 minutes during market hours",
        "comment_depth": "Top-level comments only (performance optimization)",
        "ticker_extraction": "Regex patterns for $TICKER, TICKER$, and standalone tickers",
        "sentiment_analysis": "VADER sentiment analysis on post titles and content",
        "signal_generation": "Based on mention frequency, sentiment scores, and post engagement"
    },
    "rate_limit_management": {
        "authenticated_limit": "600 requests per 10 minutes",
        "fallback_strategy": "Reduce polling frequency during high activity",
        "error_handling": "Exponential backoff for rate limit errors"
    },
    "data_processing": {
        "filtering": "Remove deleted posts, filter by minimum score/engagement",
        "deduplication": "Track processed post IDs to avoid duplicates",
        "storage": "In-memory cache with configurable retention period"
    }
}

def get_endpoint_documentation(endpoint_name: str) -> dict:
    """
    Get documentation for a specific Reddit API endpoint.

    Args:
        endpoint_name: Name of the endpoint (e.g., 'GET_hot')

    Returns:
        Dictionary containing endpoint documentation
    """
    return REDDIT_API_ENDPOINTS.get(endpoint_name, {})

def list_available_endpoints() -> list:
    """
    Get list of all documented Reddit API endpoints.

    Returns:
        List of endpoint names
    """
    return list(REDDIT_API_ENDPOINTS.keys()) + ["GET_subreddit_info"]

def get_aac_reddit_config() -> dict:
    """
    Get AAC-specific Reddit API configuration and usage patterns.

    Returns:
        Dictionary containing AAC Reddit integration details
    """
    return AAC_REDDIT_USAGE

if __name__ == "__main__":
    # Example usage
    print("AAC Reddit API Documentation")
    print("=" * 40)

    print(f"\nAvailable endpoints: {list_available_endpoints()}")

    hot_docs = get_endpoint_documentation("GET_hot")
    print(f"\nGET_hot endpoint: {hot_docs.get('endpoint', 'N/A')}")
    print(f"Description: {hot_docs.get('description', '').strip()[:100]}...")

    aac_config = get_aac_reddit_config()
    print(f"\nAAC Primary Endpoint: {aac_config.get('primary_endpoint')}")
    print(f"Primary Subreddit: r/{aac_config.get('primary_subreddit')}")