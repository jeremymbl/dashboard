from src.data_sources import get_supabase
sb = get_supabase()

# Liste rapide des tables vues par PostgREST :
tables = sb.table("pg_tables").select("schemaname,tablename").eq("schemaname", "public").execute()
print([f"{t['schemaname']}.{t['tablename']}" for t in tables.data])
