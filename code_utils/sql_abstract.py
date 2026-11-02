import re


def abstract_sql(sql: str, read: str | None = None) -> str:
    import sqlglot
    from sqlglot import exp

    parsed = sqlglot.parse_one(sql, read=read)

    def _id_name(x):
        if x is None:
            return None
        if isinstance(x, exp.Identifier):
            return x.name
        return getattr(x, "name", None) or str(x)

    table_map: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    col_map: dict[tuple[str | None, str], str] = {}
    table_counter = 0
    alias_counter = 0
    col_counter = 0

    def table_placeholder(key: str) -> str:
        nonlocal table_counter
        if key not in table_map:
            table_counter += 1
            table_map[key] = f"t{table_counter}"
        return table_map[key]

    def alias_placeholder(key: str) -> str:
        nonlocal alias_counter
        if key not in alias_map:
            alias_counter += 1
            alias_map[key] = f"a{alias_counter}"
        return alias_map[key]

    def col_placeholder(key: tuple[str | None, str]) -> str:
        nonlocal col_counter
        if key not in col_map:
            col_counter += 1
            col_map[key] = f"c{col_counter}"
        return col_map[key]

    for cte in parsed.find_all(exp.CTE):
        alias = cte.args.get("alias")
        name = _id_name(alias.this) if alias and alias.this else None
        if name:
            table_placeholder(name)

    for table in parsed.find_all(exp.Table):
        catalog = _id_name(table.args.get("catalog"))
        db = _id_name(table.args.get("db"))
        name = table.name
        key = ".".join([p for p in (catalog, db, name) if p])
        table_placeholder(key)

        alias = table.args.get("alias")
        alias_name = _id_name(alias.this) if alias and alias.this else None
        if alias_name:
            alias_placeholder(alias_name)

    def _transform(node):
        if isinstance(node, exp.CTE):
            alias = node.args.get("alias")
            name = _id_name(alias.this) if alias and alias.this else None
            if name:
                ph = table_map.get(name) or table_placeholder(name)
                node = node.copy()
                node.args["alias"].set("this", exp.to_identifier(ph))
            return node

        if isinstance(node, exp.Table):
            catalog = _id_name(node.args.get("catalog"))
            db = _id_name(node.args.get("db"))
            name = node.name
            key = ".".join([p for p in (catalog, db, name) if p])
            ph = table_map.get(key) or table_placeholder(key)

            node = node.copy()
            node.set("catalog", None)
            node.set("db", None)
            node.set("this", exp.to_identifier(ph))

            alias = node.args.get("alias")
            alias_name = _id_name(alias.this) if alias and alias.this else None
            if alias_name:
                aph = alias_map.get(alias_name) or alias_placeholder(alias_name)
                alias.set("this", exp.to_identifier(aph))

            return node

        if isinstance(node, exp.Column):
            table_id = node.args.get("table")
            qualifier = _id_name(table_id) if table_id else None
            if qualifier:
                qph = alias_map.get(qualifier) or table_map.get(qualifier)
                if not qph:
                    qph = table_placeholder(qualifier)
            else:
                qph = None

            key = (qph, node.name)
            cph = col_placeholder(key)

            node = node.copy()
            node.set("this", exp.to_identifier(cph))
            if qph:
                node.set("table", exp.to_identifier(qph))
            return node

        if isinstance(node, exp.Literal):
            if node.is_string:
                return exp.Literal.string("?")
            if node.is_number:
                return exp.Literal.number(0)
            return exp.Literal.string("?")

        if isinstance(node, exp.Placeholder):
            return exp.Placeholder(this=exp.to_identifier("p"), kind=node.args.get("kind"))

        return node

    abstracted = parsed.transform(_transform, copy=True)
    abstracted_sql = abstracted.sql(dialect="ansi")
    abstracted_sql = re.sub(r"\s+", " ", abstracted_sql).strip().rstrip(";").lower()
    return abstracted_sql

