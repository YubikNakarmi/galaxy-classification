All the alembic revisions are on `alembic_migrations/versions` folder. The `alembic.ini` file contains the database connection string and other configuration options.
## Updating migrations

``` bash
alembic revision --autogenerate -m "Added new field to User model"
almbic upgrade head
```

## Rolling back migrations

``` bash
alembic downgrade -1
```