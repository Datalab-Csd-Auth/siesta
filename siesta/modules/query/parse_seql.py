"""
responded_pairs_dsl.py
======================
DSL parser that accepts a regexp-like pattern over activity labels and
emits all ``RespondedPair`` objects to be verified against an
``ActivityPairsIndex``.

Grammar
-------
::

    pattern    ::= or_expr
    or_expr    ::= seq_expr ('||' seq_expr)*
    seq_expr   ::= element+
    element    ::= neg_atom quantifier?
    neg_atom   ::= ('!' | '^') atom | atom      # negation prefix
    quantifier ::= '*' | '+' | '?'
    atom       ::= activity | '(' or_expr ')'
    activity   ::= LABEL ('[' attr_list ']')?
    attr_list  ::= attr (',' attr)*
    attr       ::= LABEL ('=' | '!=') attr_value
    attr_value ::= STRING | var_expr
    var_expr   ::= '$' NUMBER (('+' | '-') NUMBER)?

Pair semantics
--------------
For each OR-branch (a fully resolved alternative of the pattern), every
ordered pair (A, B) of *distinct-position* **positive** activities where
A can precede B in a valid trace is emitted.  Self-pairs (A -> A) are excluded.

Negation
--------
A negated element ``!x`` (or ``^x``) marks activity ``x`` as **forbidden**
between the surrounding positive activities.  Negated activities are never
pair endpoints; instead they populate the ``forbidden_between`` set of every
pair whose positions span them::

    a !b c       -> pair (a,c) with forbidden_between={b}
    a !(b||c) d  -> pair (a,d) with forbidden_between={b,c}   # OR is unioned
    a !b c d     -> (a,c): forbidden={b}; (a,d): forbidden={b}; (c,d): forbidden={}

The CEP engine uses ``forbidden_between`` to reject candidates where any
forbidden activity appears between ``pos_a`` and ``pos_b`` in the trace.

Attribute constraints travel with each pair so that a downstream CEP
engine can validate them against ``ActivityPairsIndex`` candidates.

Example
-------
>>> pairs = extract_responded_pairs("a b*(c||d)")
>>> [(p.source.label, p.target.label) for p in pairs]
[('a', 'b'), ('a', 'c'), ('b', 'c'), ('a', 'b'), ('a', 'd'), ('b', 'd')]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Attribute-value AST
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StringLiteral:
    """A literal string attribute value, e.g. ``"nick"``."""
    value: str

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass(frozen=True)
class VarExpr:
    """
    A variable reference with an optional arithmetic offset.

    Examples::

        $1          ->  VarExpr(var_id=1)
        $2+5        ->  VarExpr(var_id=2, op='+', offset=5)
        $3-10       ->  VarExpr(var_id=3, op='-', offset=10)

    Notes
    -----
    The parser does **not** validate whether the arithmetic is semantically
    applicable - that responsibility belongs to the CEP engine.
    """
    var_id: int
    op: Optional[str] = None      # '+' | '-' | None
    offset: Optional[int] = None

    def __str__(self) -> str:
        base = f"${self.var_id}"
        if self.op is not None:
            base += f"{self.op}{self.offset}"
        return base


AttrValue = Union[StringLiteral, VarExpr]


@dataclass(frozen=True)
class AttrConstraint:
    """A single ``name (= | !=) value`` constraint inside an activity's ``[…]``."""
    name: str
    value: AttrValue
    op: str = "="          # "=" (equality) or "!=" (inequality)

    def __str__(self) -> str:
        return f"{self.name}{self.op}{self.value}"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Core pattern AST
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ActivityNode:
    """
    A single activity label, optionally annotated with attribute constraints.

    Examples::

        a
        Submit_Application[resource="nick"]
        Review[user=$1]
        Approve[amount=$2+100]
    """
    label: str
    constraints: Tuple[AttrConstraint, ...] = ()

    def __str__(self) -> str:
        if not self.constraints:
            return self.label
        cs = ", ".join(str(c) for c in self.constraints)
        return f"{self.label}[{cs}]"


class Quantifier(Enum):
    """Repetition quantifier for a pattern element."""
    ONE  = "1"   # exactly once (no symbol)
    STAR = "*"   # zero or more
    PLUS = "+"   # one or more
    OPT  = "?"   # zero or one

    def can_repeat(self) -> bool:
        """True iff the element is allowed to occur more than once."""
        return self in (Quantifier.STAR, Quantifier.PLUS)

    def can_skip(self) -> bool:
        """True iff the element is allowed to not occur at all."""
        return self in (Quantifier.STAR, Quantifier.OPT)

    def __str__(self) -> str:
        return "" if self == Quantifier.ONE else self.value


@dataclass
class ElementNode:
    """An atom (possibly negated) with an optional repetition quantifier."""
    atom: Union[ActivityNode, SeqNode, OrNode, NegatedNode]
    quantifier: Quantifier = Quantifier.ONE

    def __str__(self) -> str:
        inner = str(self.atom)
        if isinstance(self.atom, (SeqNode, OrNode)):
            inner = f"({inner})"
        return inner + str(self.quantifier)


@dataclass
class SeqNode:
    elements: List[ElementNode]

    def __str__(self) -> str:
        return "".join(str(e) for e in self.elements)


@dataclass
class OrNode:
    branches: List[SeqNode]

    def __str__(self) -> str:
        return " || ".join(str(b) for b in self.branches)


@dataclass
class NegatedNode:
    """
    A negated atom - activity or OR-group - introduced by ``!`` or ``^``.

    Semantics
    ---------
    The activities inside must **not** appear between the positive activities
    that surround this node in the sequence.  When the inner node is an
    ``OrNode``, *all* its alternatives are forbidden (OR is unioned, not
    branched).

    Examples::

        !b          -> activity b is forbidden
        ^b          -> same, caret is an alias for !
        !(b||c)     -> both b and c are forbidden (union)
        !Validate   -> activity Validate is forbidden
    """
    inner: PatternNode      # ActivityNode or OrNode of activities

    def forbidden_activities(self) -> List[ActivityNode]:
        """
        Collect every ``ActivityNode`` reachable inside this negation.

        OR-alternatives are **unioned** (no branch split): ``!(a||b)``
        yields ``[a, b]``, both forbidden in a single sequence slot.
        """
        return _collect_forbidden(self.inner)

    def __str__(self) -> str:
        inner = str(self.inner)
        if isinstance(self.inner, (SeqNode, OrNode)):
            inner = f"({inner})"
        return f"!{inner}"


def _collect_forbidden(node: PatternNode) -> List[ActivityNode]:
    """Recursively gather all ActivityNodes inside a negated sub-pattern."""
    if isinstance(node, ActivityNode):
        return [node]
    if isinstance(node, OrNode):
        result: List[ActivityNode] = []
        for branch in node.branches:
            result.extend(_collect_forbidden(branch))
        return result
    if isinstance(node, SeqNode):
        result = []
        for elem in node.elements:
            result.extend(_collect_forbidden(elem.atom))
        return result
    return []


PatternNode = Union[ActivityNode, SeqNode, OrNode, NegatedNode]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Lexer
# ═══════════════════════════════════════════════════════════════════════════

class TT(Enum):
    """Token types."""
    OR       = "OR"
    NOT      = "NOT"      # ! or ^  - negation prefix
    LABEL    = "LABEL"
    LPAREN   = "LPAREN"
    RPAREN   = "RPAREN"
    LBRACK   = "LBRACK"
    RBRACK   = "RBRACK"
    STAR     = "STAR"
    PLUS     = "PLUS"
    MINUS    = "MINUS"
    OPT      = "OPT"
    EQ       = "EQ"
    NEQ      = "NEQ"      # != - attribute inequality
    COMMA    = "COMMA"
    STRING   = "STRING"
    VAR      = "VAR"
    NUMBER   = "NUMBER"
    EOF      = "EOF"


_RAW_PATTERNS: List[Tuple[TT, str]] = [
    (TT.OR,     r'\|\|'),
    (TT.NEQ,    r'!='),          # must precede NOT so '!=' wins over a lone '!'
    (TT.NOT,    r'[!^]'),
    (TT.STRING, r'"(?:[^"\\]|\\.)*"'),
    (TT.VAR,    r'\$\d+'),
    (TT.LABEL,  r'[A-Za-z_][A-Za-z0-9_:\\]*'),
    (TT.NUMBER, r'\d+'),
    (TT.LPAREN, r'\('),
    (TT.RPAREN, r'\)'),
    (TT.LBRACK, r'\['),
    (TT.RBRACK, r'\]'),
    (TT.STAR,   r'\*'),
    (TT.PLUS,   r'\+'),
    (TT.MINUS,  r'-'),
    (TT.OPT,    r'\?'),
    (TT.EQ,     r'='),
    (TT.COMMA,  r','),
]

_MASTER_RE = re.compile(
    "|".join(
        f"(?P<G{i}>{pat})" for i, (_, pat) in enumerate(_RAW_PATTERNS)
    )
)

_UNESCAPE_RE = re.compile(r'\\(.)')


@dataclass(frozen=True)
class Token:
    type: TT
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.type.value}, {self.value!r}, @{self.pos})"


def tokenize(text: str) -> List[Token]:
    """
    Convert a DSL pattern string into a flat list of :class:`Token` objects.

    Raises
    ------
    SyntaxError
        On any character that does not match a known token pattern.
    """
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _MASTER_RE.match(text, pos)
        if not m or not m.lastgroup:
            raise SyntaxError(
                f"Unexpected character {text[pos]!r} at position {pos}"
            )
        idx = int(m.lastgroup[1:])   # "G3" -> 3
        tok_type = _RAW_PATTERNS[idx][0]
        tokens.append(Token(tok_type, m.group(), pos))
        pos = m.end()
    tokens.append(Token(TT.EOF, "", pos))
    return tokens


def split_pattern_to_list(pattern: str) -> list:
    """
    Parse *pattern* and return a flat, ordered list of every activity
    occurrence - positive and negated - as plain dictionaries.

    This is a read-only inspection utility; it does **not** resolve OR
    branches or perform pair extraction.  Each activity appears in the
    order it is written in the pattern, with its quantifier and attribute
    annotations intact.

    Parameters
    ----------
    pattern : str
        A SeQL pattern string, e.g.::

            'ACT_A[x="1", y=$1]* ACT_B[y="5"] ACT_C[y=$1+2]'

        Note: bare numeric literals (``x=1``) are not supported by the
        parser - wrap them in quotes (``x="1"``) or use a variable
        (``x=$1``).

    Returns
    -------
    list[dict]
        One dict per activity occurrence, in left-to-right pattern order.
        Each dict has the following keys:

        ``label`` : str
            The activity name (e.g. ``"ACT_A"``).
        ``quantifier`` : str or None
            ``"*"``, ``"+"``, ``"?"`` or ``None`` (exactly once).
        ``negated`` : bool
            ``True`` if the activity was preceded by ``!`` / ``^``.
        ``attributes`` : list[dict]
            One dict per ``name=value`` annotation, each with keys:

            ``name`` : str  - attribute name
            ``kind`` : ``"literal"`` | ``"variable"``
            ``value`` : str - the raw literal value (quotes stripped), or
                              the variable reference as a string
                              (e.g. ``"$1"``, ``"$2+5"``, ``"$3-10"``).

    Examples
    --------
    >>> split_pattern_to_list('ACT_A[x="1", y=$1]* ACT_B[y="5"] ACT_C[y=$1+2]')
    [
      {'label': 'ACT_A', 'quantifier': '*', 'negated': False,
       'attributes': [
         {'name': 'x', 'kind': 'literal',  'value': '1'},
         {'name': 'y', 'kind': 'variable', 'value': '$1'},
       ]},
      {'label': 'ACT_B', 'quantifier': None, 'negated': False,
       'attributes': [
         {'name': 'y', 'kind': 'literal', 'value': '5'},
       ]},
      {'label': 'ACT_C', 'quantifier': None, 'negated': False,
       'attributes': [
         {'name': 'y', 'kind': 'variable', 'value': '$1+2'},
       ]},
    ]
    """
    ast = parse_pattern(pattern)
    results: list = []
    _collect_activities(ast, negated=False, quantifier=Quantifier.ONE, out=results)
    return results


def _collect_activities(
    node,
    negated: bool,
    quantifier: "Quantifier",
    out: list,
) -> None:
    """Recursive AST walker that appends activity dicts to *out*."""

    if isinstance(node, ActivityNode):
        # Build the quantifier symbol string
        q_map = {
            Quantifier.ONE:  None,
            Quantifier.STAR: "*",
            Quantifier.PLUS: "+",
            Quantifier.OPT:  "?",
        }
        # Serialise each attribute constraint
        attrs = []
        for c in node.constraints:
            if isinstance(c.value, StringLiteral):
                attrs.append({
                    "name":  c.name,
                    "op":    c.op,                   # "=" or "!="
                    "kind":  "literal",
                    "value": c.value.value,          # quotes already stripped
                })
            elif isinstance(c.value, VarExpr):
                raw = f"${c.value.var_id}"
                if c.value.op is not None:
                    raw += f"{c.value.op}{c.value.offset}"
                attrs.append({
                    "name":  c.name,
                    "op":    c.op,                   # "=" or "!="
                    "kind":  "variable",
                    "value": raw,
                })

        out.append({
            "label":      node.label,
            "quantifier": q_map[quantifier],
            "negated":    negated,
            "attributes": attrs,
        })
        return

    if isinstance(node, NegatedNode):
        # Everything inside is negated; quantifier was already propagated
        # by the ElementNode wrapper, so pass it through unchanged.
        _collect_activities(node.inner, negated=True, quantifier=quantifier, out=out)
        return

    if isinstance(node, SeqNode):
        for elem in node.elements:
            # Merge the element's own quantifier with any inherited one
            eff_q = _combine_quantifiers(quantifier, elem.quantifier)
            _collect_activities(elem.atom, negated=negated, quantifier=eff_q, out=out)
        return

    if isinstance(node, OrNode):
        for branch in node.branches:
            _collect_activities(branch, negated=negated, quantifier=quantifier, out=out)
        return


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Parser  (recursive-descent)
# ═══════════════════════════════════════════════════════════════════════════

class Parser:
    """
    Recursive-descent parser for the DSL grammar.

    Operator precedence (high -> low)
    ---------------------------------
    1. Quantifier  (* + ?)
    2. Sequence    (concatenation)
    3. Alternation (||)
    """

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ── internal helpers ─────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _consume(self, expected: Optional[TT] = None) -> Token:
        tok = self._tokens[self._pos]
        if expected is not None and tok.type != expected:
            raise SyntaxError(
                f"Expected {expected.value!r}, got {tok.type.value!r} "
                f"({tok.value!r}) at position {tok.pos}"
            )
        self._pos += 1
        return tok

    def _at(self, *types: TT) -> bool:
        return self._peek().type in types

    # ── grammar rules ─────────────────────────────────────────────────────

    def parse(self) -> PatternNode:
        """Entry point - parse the full pattern."""
        node = self._or_expr()
        self._consume(TT.EOF)
        return node

    # or_expr ::= seq_expr ('||' seq_expr)*
    def _or_expr(self) -> PatternNode:
        branches = [self._seq_expr()]
        while self._at(TT.OR):
            self._consume(TT.OR)
            branches.append(self._seq_expr())
        return branches[0] if len(branches) == 1 else OrNode(branches)

    # seq_expr ::= element+
    def _seq_expr(self) -> SeqNode:
        elements: List[ElementNode] = []
        while not self._at(TT.RPAREN, TT.OR, TT.EOF):
            elements.append(self._element())
        if not elements:
            raise SyntaxError(
                f"Empty sequence at position {self._peek().pos}"
            )
        return SeqNode(elements)

    # element  ::= neg_atom quantifier?
    # neg_atom ::= ('!' | '^') atom | atom
    def _element(self) -> ElementNode:
        negated = False
        if self._at(TT.NOT):
            self._consume(TT.NOT)
            negated = True
        atom = self._atom()
        if negated:
            atom = NegatedNode(inner=atom)
        quant = Quantifier.ONE
        if   self._at(TT.STAR):  self._consume(); quant = Quantifier.STAR
        elif self._at(TT.PLUS):  self._consume(); quant = Quantifier.PLUS
        elif self._at(TT.OPT):   self._consume(); quant = Quantifier.OPT
        return ElementNode(atom=atom, quantifier=quant)

    # atom ::= activity | '(' or_expr ')'
    def _atom(self) -> PatternNode:
        if self._at(TT.LPAREN):
            self._consume(TT.LPAREN)
            inner = self._or_expr()
            self._consume(TT.RPAREN)
            return inner
        return self._activity()

    # activity ::= LABEL ('[' attr_list ']')?
    def _activity(self) -> ActivityNode:
        if self._at(TT.STRING):          # <- quoted label branch
            tok = self._consume(TT.STRING)
            label = _UNESCAPE_RE.sub(r'\1', tok.value[1:-1])  # strip quotes + unescape
        else:
            label = self._consume(TT.LABEL).value
        constraints: List[AttrConstraint] = []
        if self._at(TT.LBRACK):
            self._consume(TT.LBRACK)
            constraints = self._attr_list()
            self._consume(TT.RBRACK)
        return ActivityNode(label=label, constraints=tuple(constraints))

    # attr_list ::= attr (',' attr)*
    # attr      ::= LABEL ('=' | '!=') attr_value
    def _attr_list(self) -> List[AttrConstraint]:
        attrs: List[AttrConstraint] = []
        while True:
            name = self._consume(TT.LABEL).value
            if self._at(TT.NEQ):
                self._consume(TT.NEQ)
                op = "!="
            else:
                self._consume(TT.EQ)
                op = "="
            value = self._attr_value()
            attrs.append(AttrConstraint(name=name, value=value, op=op))
            if not self._at(TT.COMMA):
                break
            self._consume(TT.COMMA)
        return attrs

    # attr_value ::= STRING | var_expr
    # var_expr   ::= '$' NUMBER (('+' | '-') NUMBER)?
    def _attr_value(self) -> AttrValue:
        if self._at(TT.STRING):
            tok = self._consume(TT.STRING)
            return StringLiteral(tok.value[1:-1])   # strip surrounding quotes
        if self._at(TT.VAR):
            tok = self._consume(TT.VAR)
            var_id = int(tok.value[1:])
            op: Optional[str] = None
            offset: Optional[int] = None
            # Inside [...], a '+' or '-' after $N is arithmetic, not a quantifier
            if self._at(TT.PLUS, TT.MINUS):
                op = self._consume().value
                offset = int(self._consume(TT.NUMBER).value)
            return VarExpr(var_id=var_id, op=op, offset=offset)
        raise SyntaxError(
            f"Expected attribute value (string or $var) at position "
            f"{self._peek().pos}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Lineariser - OR resolution via cartesian product
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundActivity:
    """
    An activity as it appears in a fully-linearised branch, carrying the
    *effective* quantifier inherited from all enclosing group quantifiers,
    and a flag indicating whether it is a **negation** (forbidden) marker.
    """
    activity: ActivityNode
    quantifier: Quantifier
    negated: bool = False          # True  -> forbidden-between marker

    def __repr__(self) -> str:
        prefix = "!" if self.negated else ""
        return f"{prefix}{self.activity}{self.quantifier}"


def _combine_quantifiers(inner: Quantifier, outer: Quantifier) -> Quantifier:
    """
    Merge quantifiers when a group inherits one from its parent element.

    The quantifier that is *most permissive* (allows the most executions)
    dominates::

        STAR > PLUS > OPT > ONE
    """
    order = {
        Quantifier.ONE:  0,
        Quantifier.OPT:  1,
        Quantifier.PLUS: 2,
        Quantifier.STAR: 3,
    }
    return outer if order[outer] > order[inner] else inner


def _linearise(node: PatternNode) -> List[List[BoundActivity]]:
    """
    Expand the AST into a list of *linear sequences*, one per OR-branch.

    ``OrNode`` causes a split (union of children's sequences).
    ``SeqNode`` causes a cartesian product of its elements' sequences.
    ``ActivityNode`` returns a singleton sequence.
    ``NegatedNode`` returns a single sequence of forbidden-marked activities
      (OR inside a negation is **unioned**, not branched).

    Returns
    -------
    List[List[BoundActivity]]
        Each inner list is one fully-resolved linear sequence.
    """
    if isinstance(node, ActivityNode):
        return [[BoundActivity(node, Quantifier.ONE, negated=False)]]

    if isinstance(node, NegatedNode):
        # Collect all forbidden activities without branching on OR
        forbidden = node.forbidden_activities()
        return [[BoundActivity(a, Quantifier.ONE, negated=True) for a in forbidden]]

    if isinstance(node, OrNode):
        result: List[List[BoundActivity]] = []
        for branch in node.branches:
            result.extend(_linearise(branch))
        return result

    if isinstance(node, SeqNode):
        sequences: List[List[BoundActivity]] = [[]]
        for elem in node.elements:
            elem_alts = _linearise_element(elem)
            sequences = [
                prefix + suffix
                for prefix in sequences
                for suffix in elem_alts
            ]
        return sequences

    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


def _linearise_element(elem: ElementNode) -> List[List[BoundActivity]]:
    """
    Linearise an element, propagating its quantifier to every contained
    positive activity.  Negated activities keep their ``negated=True`` flag
    but also receive the quantifier (the CEP engine may ignore it).
    """
    atom_alts = _linearise(elem.atom)
    return [
        [
            BoundActivity(
                activity=ba.activity,
                quantifier=_combine_quantifiers(ba.quantifier, elem.quantifier),
                negated=ba.negated,
            )
            for ba in alt
        ]
        for alt in atom_alts
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Pair extraction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RespondedPair:
    """
    A responded-existence pair ``(source -> target)`` derived from the pattern.

    Attributes
    ----------
    source : ActivityNode
        The leading activity, complete with its attribute constraints.
    target : ActivityNode
        The following activity, complete with its attribute constraints.
    source_quantifier : Quantifier
        Effective repetition quantifier of the source in its branch.
    target_quantifier : Quantifier
        Effective repetition quantifier of the target in its branch.
    forbidden_between : Tuple[ActivityNode, ...]
        Activities that must **not** appear between ``source`` and ``target``
        in a matching trace.  Populated by ``!x`` / ``^x`` elements that
        sit at positions strictly between source and target in the pattern.
        The CEP engine is responsible for enforcing this constraint against
        the ``ActivityPairsIndex`` candidates.
    branch_id : int
        0-based index of the OR-branch this pair originated from.
        Pairs from different branches may share the same label-pair but
        carry different attribute or negation constraint contexts.

    Usage with ActivityPairsIndex
    ------------------------------
    ::

        candidates = index.get(pair.key, [])
        # candidates: [(trace_id, pos_a, pos_b, attrs_a, attrs_b), ...]
        # Forward pair + candidates to the CEP engine for full validation:
        #   - attribute constraints (source.constraints, target.constraints)
        #   - forbidden_between: no activity in this set may appear at
        #     any position k with pos_a < k < pos_b in the trace
    """
    source: ActivityNode
    target: ActivityNode
    source_quantifier: Quantifier
    target_quantifier: Quantifier
    forbidden_between: Tuple[ActivityNode, ...]
    branch_id: int

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def key(self) -> Tuple[str, str]:
        """Label-only key for ``ActivityPairsIndex`` lookup."""
        return (self.source.label, self.target.label)

    def short(self) -> str:
        """Compact label-only representation (``ab``, ``ac``, ...)."""
        return f"{self.source.label}{self.target.label}"

    def forbidden_labels(self) -> Tuple[str, ...]:
        """Label-only set of forbidden-between activities."""
        return tuple(a.label for a in self.forbidden_between)

    def __str__(self) -> str:
        sq = "" if self.source_quantifier == Quantifier.ONE else str(self.source_quantifier)
        tq = "" if self.target_quantifier == Quantifier.ONE else str(self.target_quantifier)
        fb = ""
        if self.forbidden_between:
            labels = ", ".join(a.label for a in self.forbidden_between)
            fb = f"  forbidden={{{labels}}}"
        return (
            f"({self.source}{sq} -> {self.target}{tq})"
            f"  [branch={self.branch_id}]{fb}"
        )

    def __repr__(self) -> str:
        return (
            f"RespondedPair(source={self.source!r}, target={self.target!r}, "
            f"src_q={self.source_quantifier.value}, "
            f"tgt_q={self.target_quantifier.value}, "
            f"forbidden={self.forbidden_labels()}, "
            f"branch={self.branch_id})"
        )

    def __eq__(self, other):
        """Compare to check equality"""
        return (
            # instance of the same class
            isinstance(other, RespondedPair)
            and self.source == other.source
            and self.target == other.target
        )

    def __hash__(self):
        """Generate hash value for this instance"""
        return hash((self.source, self.target))

def _pairs_from_sequence(
    seq: List[BoundActivity],
    branch_id: int,
) -> List[RespondedPair]:
    """
    Generate all ordered, non-self responded pairs from one linear sequence.

    Only **positive** (non-negated) activities can be pair endpoints.
    For each pair of positive positions ``i < j``, every negated
    ``BoundActivity`` at a position strictly between ``i`` and ``j``
    contributes to the pair's ``forbidden_between`` set.
    """
    pairs: List[RespondedPair] = []
    # Only positive activities are pair endpoints
    pos_indices = [k for k, ba in enumerate(seq) if not ba.negated]
    n = len(pos_indices)
    for ii in range(n):
        if seq[pos_indices[ii]].quantifier in (Quantifier.STAR, Quantifier.PLUS) and False:
            dup_pair = seq[pos_indices[ii]]
            pairs.append(
                RespondedPair(
                    source=dup_pair.activity,
                    target=dup_pair.activity,
                    source_quantifier=dup_pair.quantifier,
                    target_quantifier=dup_pair.quantifier,
                    forbidden_between=(),
                    branch_id=branch_id,
                )
            )
        for jj in range(ii + 1, n):
            i, j = pos_indices[ii], pos_indices[jj]
            # Collect negated activities strictly between i and j
            forbidden: Tuple[ActivityNode, ...] = tuple(
                seq[k].activity
                for k in range(i + 1, j)
                if seq[k].negated
            )
            pairs.append(
                RespondedPair(
                    source=seq[i].activity,
                    target=seq[j].activity,
                    source_quantifier=seq[i].quantifier,
                    target_quantifier=seq[j].quantifier,
                    forbidden_between=forbidden,
                    branch_id=branch_id,
                )
            )
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Public API
# ═══════════════════════════════════════════════════════════════════════════

_COMPACT_RE = re.compile(
    r"""
    (?P<word>[A-Za-z_][A-Za-z0-9_]*)  # existing label (greedy)
    |(?P<sym>[^A-Za-z0-9_])            # any non-label character
    """,
    re.VERBOSE,
)


def expand_compact(pattern: str) -> str:
    """
    Expand a compact single-character pattern into a space-separated one.

    In compact notation each alphabetic character is treated as a
    *separate* activity, making ``"ab*(c||d)"`` equivalent to
    ``"a b*(c||d)"``.  This is convenient for quick examples and tests
    but **must not** be used when activity labels are multi-character
    (e.g. ``"Submit_Application"``).

    The expansion is character-level: it inserts a space between every
    pair of consecutive label characters.  Attribute blocks ``[…]`` and
    all operator characters are preserved unchanged.

    Examples
    --------
    >>> expand_compact("ab*(c||d)")
    'a b*(c||d)'
    >>> expand_compact("a[res=$1]b*c")   # brackets already separate labels
    'a[res=$1]b*c'                        # no change needed; space harmless
    """
    tokens = tokenize(pattern)
    parts: List[str] = []
    prev_was_label = False
    for tok in tokens:
        if tok.type == TT.EOF:
            break
        if tok.type == TT.LABEL:
            if prev_was_label:
                # Insert a space to force separate activity tokens
                for ch in tok.value:
                    parts.append(" " + ch)
            else:
                # Emit first char, rest as space-separated
                chars = list(tok.value)
                parts.append(chars[0])
                for ch in chars[1:]:
                    parts.append(" " + ch)
            prev_was_label = True
        else:
            parts.append(tok.value)
            prev_was_label = False
    return "".join(parts)


def parse_pattern(pattern: str) -> PatternNode:
    """
    Parse a DSL pattern string and return the root AST node.

    Parameters
    ----------
    pattern : str
        A DSL pattern such as ``'ab*(c||d)'``.

    Returns
    -------
    PatternNode
        Root of the AST (``SeqNode``, ``OrNode``, or ``ActivityNode``).

    Raises
    ------
    SyntaxError
        On any tokenisation or grammar error.
    """
    return Parser(tokenize(pattern)).parse()


def extract_responded_pairs(pattern: str) -> List[RespondedPair]:
    """
    Parse *pattern* and return all responded pairs.

    OR-alternatives produce separate branches; the pairs from all branches
    are concatenated in branch order.  The same label-pair may appear in
    multiple branches with different attribute constraint contexts - this is
    intentional and must not be de-duplicated at this stage.

    .. note::
        **Activity-label parsing** - because this DSL supports multi-character
        labels (e.g. ``Submit_Application``), consecutive letters are treated
        as a *single* label.  Separate activities must be delimited by
        whitespace or by their ``[…]`` attribute blocks:

        =====================  ============================
        Intent                 Correct pattern syntax
        =====================  ============================
        label ``ab`` (one act) ``ab*(c||d)``
        labels a, b (two acts) ``a b*(c||d)``
        =====================  ============================

        Use :func:`expand_compact` to auto-expand single-character shorthands.

    Parameters
    ----------
    pattern : str
        DSL pattern, e.g.::

            'a b*(c||d)'
            'a[resource="nick"] b*c'
            'a[resource=$1] b[resource=$1]*c'
            'a[amount=$1] b[amount=$1+5]*(c||d)'

    Returns
    -------
    List[RespondedPair]
        Pairs in branch-first, left-to-right positional order.

    Examples
    --------
    >>> pairs = extract_responded_pairs("a b*(c||d)")
    >>> [p.short() for p in pairs]
    ['ab', 'ac', 'bc', 'ab', 'ad', 'bd']

    >>> # Compact single-char shorthand (auto-expand first):
    >>> pairs = extract_responded_pairs(expand_compact("ab*(c||d)"))
    >>> [p.short() for p in pairs]
    ['ab', 'ac', 'bc', 'ab', 'ad', 'bd']
    """
    ast = parse_pattern(pattern)
    sequences = _linearise(ast)
    result: List[RespondedPair] = []
    for branch_id, seq in enumerate(sequences):
        result.extend(_pairs_from_sequence(seq, branch_id))
    return result
    results = [_pairs_from_sequence(seq, branch_id) for branch_id, seq in enumerate(sequences)]
    return results

def extract_info_pairs(pattern):
    """
    Adding self pairs for attributes or negation to better inform CEP
    """
    ast = parse_pattern(pattern)
    sequences = _linearise(ast)
    attribute_pairs = set()
    for branch_id, seq in enumerate(sequences):
        for idx, boundActivity in enumerate(seq):
            if boundActivity.activity.constraints or boundActivity.negated or boundActivity.quantifier in (Quantifier.PLUS, Quantifier.STAR):
                attribute_pairs.add((boundActivity.activity.label, boundActivity.activity.label, branch_id))
            if boundActivity.negated:
                # Find nearest non-negated predecessor
                prev_positive = next(
                    (seq[k] for k in range(idx - 1, -1, -1) if not seq[k].negated),
                    None,
                )
                # Find nearest non-negated successor
                next_positive = next(
                    (seq[k] for k in range(idx + 1, len(seq)) if not seq[k].negated),
                    None,
                )
                if prev_positive is not None:
                    attribute_pairs.add((prev_positive.activity.label, boundActivity.activity.label))
                elif next_positive is not None:
                    # No positive predecessor - anchor to the next positive event instead
                    attribute_pairs.add((boundActivity.activity.label, next_positive.activity.label))
    
    return attribute_pairs
    

def extract_siesta_pairs(pattern: str) -> List[List[BoundActivity]]:
    ast = parse_pattern(pattern)
    sequences = _linearise(ast)

    return sequences


# ═══════════════════════════════════════════════════════════════════════════
# 8.  ActivityPairsIndex query helper
# ═══════════════════════════════════════════════════════════════════════════

TraceId  = str
Position = int
Attrs    = Dict[str, object]

# A single row from the index for a given (label_a, label_b) key
IndexRow = Tuple[TraceId, Position, Position, Attrs, Attrs]

# The full index structure
ActivityPairsIndex = Dict[Tuple[str, str], List[IndexRow]]


@dataclass
class QueryResult:
    """
    The result of looking up one :class:`RespondedPair` in the index.

    ``candidates`` holds all raw index rows for the label-pair; attribute
    constraint checking is delegated to the CEP engine.
    """
    pair: RespondedPair
    candidates: List[IndexRow]

    def __repr__(self) -> str:
        return (
            f"QueryResult(pair={self.pair.short()!r}, "
            f"branch={self.pair.branch_id}, "
            f"candidates={len(self.candidates)})"
        )


def query_index(
    pairs: List[RespondedPair],
    index: ActivityPairsIndex,
) -> List[QueryResult]:
    """
    Perform label-based lookups for every responded pair.

    Attribute constraints are **not** evaluated here; the raw
    ``IndexRow`` candidates are returned verbatim for the CEP engine.

    Parameters
    ----------
    pairs : List[RespondedPair]
        Output of :func:`extract_responded_pairs`.
    index : ActivityPairsIndex
        Maps ``(label_a, label_b)`` ->
        ``[(trace_id, pos_a, pos_b, attrs_a, attrs_b), …]``.

    Returns
    -------
    List[QueryResult]
        One entry per pair (same order), with its candidate index rows.
    """
    return [
        QueryResult(pair=pair, candidates=index.get(pair.key, []))
        for pair in pairs
    ]



# # ═══════════════════════════════════════════════════════════════════════════
# # 9.  CLI demo
# # ═══════════════════════════════════════════════════════════════════════════

# def _demo() -> None:
#     separator = "─" * 64

#     examples = [
#         # (pattern, description, use_compact)
#         # (
#         #     'abcd[amount=$\"15\"]',
#         #     'example',
#         #     True
#         # ),
#         (
#             'a[amount=$1]b[amount=$1+5]*(c||d)',
#             "Variable with arithmetic offset + OR",
#             False,
#         ),
#         # ── negation examples ─────────────────────────────────────────
#         (
#             'a !b c',
#             "Negation: a -> c with b forbidden between",
#             False,
#         ),
#     ]

#     for raw_pattern, desc, use_compact in examples:
#         expanded = expand_compact(raw_pattern) if use_compact else raw_pattern
#         print(f"\n{separator}")
#         print(f"  Pattern : {raw_pattern}")
#         if use_compact:
#             print(f"  Expanded: {expanded}")
#         print(f"  Desc    : {desc}")
#         print(separator)

#         try:
#             pairs = extract_responded_pairs(expanded)
#         except SyntaxError as exc:
#             print(f"  ERROR: {exc}")
#             continue

#         branch_ids = sorted({p.branch_id for p in pairs})
#         print(f"  Branches: {len(branch_ids)}")

#         for bid in branch_ids:
#             branch_pairs = [p for p in pairs if p.branch_id == bid]
#             shorts = ", ".join(p.short() for p in branch_pairs)
#             print(f"\n  Branch {bid}: {shorts}")
#             for p in branch_pairs:
#                 fb = (
#                     f"  !! forbidden={{{', '.join(p.forbidden_labels())}}}"
#                     if p.forbidden_between else ""
#                 )
#                 src_c = (
#                     f"  src_attrs={list(p.source.constraints)}"
#                     if p.source.constraints else ""
#                 )
#                 tgt_c = (
#                     f"  tgt_attrs={list(p.target.constraints)}"
#                     if p.target.constraints else ""
#                 )
#                 print(
#                     f"    ({p.source.label}{p.source_quantifier}"
#                     f" -> {p.target.label}{p.target_quantifier})"
#                     f"{fb}{src_c}{tgt_c}"
#                 )

#     # ── ActivityPairsIndex query demo ─────────────────────────────────────
#     print(f"\n{separator}")
#     print("  ActivityPairsIndex query demo  ->  pattern: a !b (c||d)")
#     print(separator)

#     sample_index: ActivityPairsIndex = {
#         ("a", "c"): [
#             ("trace1", 0, 3, {}, {}),
#             ("trace2", 0, 2, {}, {}),
#         ],
#         ("a", "d"): [
#             ("trace3", 0, 4, {}, {}),
#         ],
#     }

#     pairs = extract_responded_pairs("a !b (c||d)")
#     results = query_index(pairs, sample_index)

#     for qr in results:
#         fb_note = (
#             f"  [forbidden between: {', '.join(qr.pair.forbidden_labels())}]"
#             if qr.pair.forbidden_between else ""
#         )
#         print(
#             f"  {qr.pair.short()!r:5s}  branch={qr.pair.branch_id}"
#             f"  -> {len(qr.candidates)} candidate(s){fb_note}"
#         )
#         for row in qr.candidates:
#             trace_id, pos_a, pos_b, attrs_a, attrs_b = row
#             print(
#                 f"         trace={trace_id!r}  pos=({pos_a},{pos_b})"
#                 f"  [CEP must verify 'b' absent in ({pos_a},{pos_b})]"
#             )


# if __name__ == "__main__":
#     _demo()