create table if not exists transactions (
  id uuid primary key,
  client_id text not null,
  user_id text not null,
  ts timestamptz not null,
  amount numeric not null,
  currency text not null default 'USD',
  merchant text,
  payment_method text,
  latitude double precision,
  longitude double precision,
  home_lat double precision,
  home_lon double precision,
  created_at timestamptz default now()
);

create index if not exists idx_transactions_client_created
  on transactions(client_id, created_at desc);

create table if not exists predictions (
  id bigserial primary key,
  client_id text not null,
  transaction_id uuid references transactions(id) on delete cascade,
  model_name text not null,
  model_version text not null,

  -- legacy fields (kept nullable for backward compatibility)
  score double precision,
  is_flagged boolean,

  -- ML v2 fields
  reconstruction_error double precision,
  anomaly_threshold double precision,
  is_predicted_anomaly boolean,

  created_at timestamptz default now(),

  unique (client_id, transaction_id, model_name, model_version)
);

create index if not exists idx_predictions_client_created
  on predictions(client_id, created_at desc);

create table if not exists alerts (
  id bigserial primary key,
  client_id text not null,
  transaction_id uuid not null references transactions(id) on delete cascade,
  status text not null default 'OPEN',
  note text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique (client_id, transaction_id)
);

create index if not exists idx_alerts_client_updated
  on alerts(client_id, updated_at desc);

create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_alerts_updated_at on alerts;

create trigger trg_alerts_updated_at
before update on alerts
for each row
execute function set_updated_at();
