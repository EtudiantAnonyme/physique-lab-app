from supabase import create_client, Client

SUPABASE_URL = "https://hiecbvryijcdrzrxbmry.supabase.co"
SUPABASE_KEY = "sb_publishable_Nhfd3EhwkxmI2JWfadRkdg_DasaDxNQ"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)