known_hashes = {
    ("ChaCha", "seed"): {
        "random_values": "0dda65c819197b3ca467f053869aa1f4a4123cb2e55a290453651916dd7002bd",
        "initial_state_hash": "1ded27d6bb53619c8b6d5decfac0e50bae731c6c962b7dafdb89c5ca941e631d",
        "final_state_hash": "9042f89dd26a7fc4c8f23ba841b61c7d7cf3a53ded37d0a61709da7b3c131ff7",
    },
    ("ChaCha", "key"): {
        "random_values": "cf07b4d5286923c2fb4318d5a985b9ba29aac59f7eb5be7ef0395c76cb5a8e71",
        "initial_state_hash": "8cc40691e74bcdef396b947c94bdb56bc965662c54afccd9da286f30842e7e4d",
        "final_state_hash": "45ecbbcc4a9b6cb1e39d90c3d0f87958ef9fcfb87574e454656745fe2c7f993b",
    },
    ("ChaCha", "seed", "counter"): {
        "random_values": "bc6eb9280caf3ebacdd7fe4632213320671774af919d310c1e5529b4dea62202",
        "initial_state_hash": "3c4127efbde9495984ecf84f5e7de6408a3b6f9e085cd16e5553320c19dc9b35",
        "final_state_hash": "e2f11faef5c70ca32a67763f307651c94162647c10aff3b3c3323c921342daa5",
    },
    ("ChaCha", "counter", "key"): {
        "random_values": "8e773b7ff72ef03fe692bd7e9fcfd0b03830d430309204e1319ed4033c9b4f2d",
        "initial_state_hash": "1e2d43ed43dbebbe9fb0dc0ab6c1d4a9e4004dac5d95340a9e0f039f64b1f4b7",
        "final_state_hash": "291e79b08b208006138169258eae8b4ab30c900b767e9f36b2885e6de4554d60",
    },
    ("Xoroshiro128", "seed"): {
        "random_values": "3ff37d80a806de5f21fb31d05897caa5e59a186d7aca1b81efc5c1ba53cfa64b",
        "initial_state_hash": "df6e73b6bc302d13b9309adef7fb207cc5e294e88883712f4b64b9c817a962f4",
        "final_state_hash": "6f706ac8c3d50298d34e0211983cee4b5bf1c41f9bc3f31374195736c595c6e9",
    },
    ("Xoroshiro128", "seed", "plusplus", True): {
        "random_values": "8b659c6525a6a050ad91c327d27817bb95dda54af3fc21de27756c2bcf2d96cb",
        "initial_state_hash": "bff56c72d4e8d24187658b8d51b178061a0a189c5f26afb6132069900e462f73",
        "final_state_hash": "087b5f13e4ef2171c5b4215aea56bbb68af23db7e0832da3f4be67506efa9a03",
    },
    ("Xoroshiro128", "seed", "plusplus", False): {
        "random_values": "3ff37d80a806de5f21fb31d05897caa5e59a186d7aca1b81efc5c1ba53cfa64b",
        "initial_state_hash": "df6e73b6bc302d13b9309adef7fb207cc5e294e88883712f4b64b9c817a962f4",
        "final_state_hash": "6f706ac8c3d50298d34e0211983cee4b5bf1c41f9bc3f31374195736c595c6e9",
    },
    ("Xorshift1024", "seed"): {
        "random_values": "1512c88b31d595bf768dd5cd5ba516a8df4a3ddfbe7251267418e861ccd1eeae",
        "initial_state_hash": "6e31fb7fe4c3abf1a136014d973b33f06f188d0f1444467279e1ed7d9954d149",
        "final_state_hash": "ca9d8ca020ad0dbbc731256a04ad50368506a5487ed62244494283da9ba723f6",
    },
    ("Xoshiro256", "seed"): {
        "random_values": "67b529922570a33ee50716624e38b4b098e2bbf18f18e4f0361805409a5d2038",
        "initial_state_hash": "fcee5f7c91d6ec387736c7281d2c2a99db21fa093fceec153b0e588ae4346540",
        "final_state_hash": "9d4caf0acb1c4eb990fe8dc91db6a38b663f7255fe0be95f5f62e577488e6410",
    },
    ("Xoshiro512", "seed"): {
        "random_values": "462d5ed83a4e6f4373291884858b66eced70b3377cabe01029ff62abfaf5d157",
        "initial_state_hash": "d2610c1fda0815d51dba75bc0f2cd627b42b418915753df477c008abb7b7ceb1",
        "final_state_hash": "0150dbdde7145328375ef9f93bbd6239dfcee15c68247d84520dc88543a217a2",
    },
    ("LXM", "seed"): {
        "random_values": "c98015d91fe08823720c58803ed790230c6d1a499367cf101d934890063da35c",
        "initial_state_hash": "627e7dcc2cef199787f262fae45ecf73a7561a0e236e7ac84db7b55a20e4f343",
        "final_state_hash": "da5473e62d244c4808cefd3818d98cb11a7e6fe65ac4bbbb7a323612abd75cb4",
    },
    ("LXM", "seed", "mix", True): {
        "random_values": "c98015d91fe08823720c58803ed790230c6d1a499367cf101d934890063da35c",
        "initial_state_hash": "627e7dcc2cef199787f262fae45ecf73a7561a0e236e7ac84db7b55a20e4f343",
        "final_state_hash": "da5473e62d244c4808cefd3818d98cb11a7e6fe65ac4bbbb7a323612abd75cb4",
    },
    ("LXM", "seed", "mix", False): {
        "random_values": "b63a8d75a689d604a4578f7e11521c05f1b88c52454cf23cdae455e95aa077d0",
        "initial_state_hash": "627e7dcc2cef199787f262fae45ecf73a7561a0e236e7ac84db7b55a20e4f343",
        "final_state_hash": "da5473e62d244c4808cefd3818d98cb11a7e6fe65ac4bbbb7a323612abd75cb4",
    },
    ("MT64", "seed"): {
        "random_values": "9c10fe3d07af95ee3c01e47802704f56c66d5d4fe80df09de9ada20da14c364c",
        "initial_state_hash": "00065ceea3c768fe4242085e5a3fc428c9f00e7bd32e5a213bfd9976271d23b2",
        "final_state_hash": "4b365cb9948e87b3892a9783c2769b7f03b4da223659e3bc63f77779ff0caa2c",
    },
    ("MT19937", "seed"): {
        "random_values": "de6797f1e0209e9a3afb9cdbe52fb524fb7fb23416e2a0aa92625319d3b4f7e8",
        "initial_state_hash": "567c1b44adca76c5c52d86fbfae48b3482b3fb6f3daa70538dd92e2d064e6317",
        "final_state_hash": "d9f8ad8a2e45cfd7ad762ce595894e804e9a0c6d27bc15070308e84a12479792",
    },
    ("DSFMT", "seed"): {
        "random_values": "d758b303c4b920998fc51510ad47e2e3e07da050cd994d47b3c5bf724749452c",
        "initial_state_hash": "b7664ff72a444ed7d65d38041f055ecbc0d098c2bac52e77ee357cde4eda24eb",
        "final_state_hash": "265fd07fa1b64d9f3f4c5011949edb496619f42ce09bdbec7aa81c56552751b5",
    },
    ("SFMT", "seed"): {
        "random_values": "57a9772542955f750bc5e08946712b14b7efdb0e0d9dd6ca9015db02b0feb5c2",
        "initial_state_hash": "efc48ff9e4c9112f4c0e960e917b960c4a607451eac36fd7582173c616cfc129",
        "final_state_hash": "56e108d5d5f5a740fde8db5b4084d5da3c9cdc8cc5a65d19f6d97512ae44534d",
    },
    ("SPECK128", "seed"): {
        "random_values": "d0a22bd5875d5c4c2262b51f06237307eb1e8b7627377c97868ce7d8d8e4f636",
        "initial_state_hash": "df6b8e5c64c036d440c7bf2574c9d898e5693d2022d6b6991d3821773c8cf1d0",
        "final_state_hash": "40029ea2116f45130388f3bdb37ed9bf12a06c7acd61a3d42e5c50c0a3f38819",
    },
    ("SPECK128", "key"): {
        "random_values": "bbba46e041a0a3b930fde1aba620a91e524d07dd3f02da94f26ecd8cd304362e",
        "initial_state_hash": "14bf2299554d57500bd8962439202a736ba79e130bd8e18d7890d18b3f55ef33",
        "final_state_hash": "2b82db1eaf2de29dfa3ed91e91a10bc05c5f57d35e0412db71b317674341d868",
    },
    ("SPECK128", "seed", "counter"): {
        "random_values": "52a5b231552b3f41b5dec43b0319aec730e89963e02a3fc154c3ddaa554085ad",
        "initial_state_hash": "2f7cae1c97125dcc846df57fae9ada86519e211d3c4259bf28473cb89285539f",
        "final_state_hash": "ecc5fefecd45b7df7917f1af3582b0e9452ba855e193f3249e97be1b02727c18",
    },
    ("SPECK128", "counter", "key"): {
        "random_values": "7a608f5e3292d6598728016cd143ec01f9fdc16b8d08824953f8c463c1e28f95",
        "initial_state_hash": "7730f500ce39885dcc24c69123763d083361f42a50aead3e5093186160900ba5",
        "final_state_hash": "78826431a76ffc95f68af869cbd38bd35be8fd7c08e43a8761110d72fba73b22",
    },
    ("PCG32", "seed"): {
        "random_values": "961142eb8786582f48a942bec78e8b059c7e0feed6019638ec804d2a0ad7e23f",
        "initial_state_hash": "246358a9ecc83746473c8be8d25de59e4426be050c553f71b76fc17c6f2448d4",
        "final_state_hash": "ce7621b9f707df3f2b9b42c324e532634703ea16ad6421f632030166f6df97fd",
    },
    ("PCG32", "seed", "inc"): {
        "random_values": "03d397920cbdccc03ebc5b6a49fdf749150a08a4e36ddd2f0cf4557b48338b7f",
        "initial_state_hash": "e4388eae288c344c78ad47cdbfe0983778d75b56c117cbb16ffdd0efc68d61aa",
        "final_state_hash": "d1b7623dfdb1084149c34472f63f65b2ae59b5013c5e72d47160597a0fe855a4",
    },
    ("PCG64", "seed"): {
        "random_values": "4d8bbff64c61c5f3158a043946b78d9c556091df421baf1dd5fbd506d7610dd2",
        "initial_state_hash": "52424eb4d0f3518635c0d36a93637c39a9d2b8f949f6dd5d737ca9b83610c3da",
        "final_state_hash": "6e7714d72766dc092faa6d2bf841aa8e8eae8530834d3eb5324b949d1740fd85",
    },
    ("PCG64", "seed", "inc"): {
        "random_values": "63fd3334317fb8902e5fdc9baf11b01ed5d098b3cdb7333e0da529edb8e09ae0",
        "initial_state_hash": "f2c4eba60d5c1dc2818987ad19994cce76fdfa1fc2680e3bd3be2012e32fd0d4",
        "final_state_hash": "e93bed1d80626ae63762c5d64ab099e321c838beaa7ee24006ad50981a0a65ed",
    },
    ("PCG64", "seed", "variant", "xsl-rr"): {
        "random_values": "4d8bbff64c61c5f3158a043946b78d9c556091df421baf1dd5fbd506d7610dd2",
        "initial_state_hash": "52424eb4d0f3518635c0d36a93637c39a9d2b8f949f6dd5d737ca9b83610c3da",
        "final_state_hash": "6e7714d72766dc092faa6d2bf841aa8e8eae8530834d3eb5324b949d1740fd85",
    },
    ("PCG64", "seed", "variant", "dxsm"): {
        "random_values": "007aaf954de7435d87abe06fb957cbade21732c6ceafb80335e89a5ca322440e",
        "initial_state_hash": "52424eb4d0f3518635c0d36a93637c39a9d2b8f949f6dd5d737ca9b83610c3da",
        "final_state_hash": "6e7714d72766dc092faa6d2bf841aa8e8eae8530834d3eb5324b949d1740fd85",
    },
    ("PCG64", "seed", "variant", "cm-dxsm"): {
        "random_values": "4b984c9ab74ab3473895b0ff4f782826f8c86b8e92e57c198440794153facf9f",
        "initial_state_hash": "3e093b9322aae9f40219284443010226a068fb2b6f42d41e83a9ee2f992c9886",
        "final_state_hash": "1aa7f247ad9245d611276469125b2b0dc25b3aed3d9ab6158a439efe0cb9155a",
    },
    ("PCG64", "seed", "inc", "variant", "xsl-rr"): {
        "random_values": "63fd3334317fb8902e5fdc9baf11b01ed5d098b3cdb7333e0da529edb8e09ae0",
        "initial_state_hash": "f2c4eba60d5c1dc2818987ad19994cce76fdfa1fc2680e3bd3be2012e32fd0d4",
        "final_state_hash": "e93bed1d80626ae63762c5d64ab099e321c838beaa7ee24006ad50981a0a65ed",
    },
    ("PCG64", "seed", "inc", "variant", "dxsm"): {
        "random_values": "88539e900241b39371085897fc069e0582ff4529876fba2f4c15d9861d0fed9b",
        "initial_state_hash": "f2c4eba60d5c1dc2818987ad19994cce76fdfa1fc2680e3bd3be2012e32fd0d4",
        "final_state_hash": "e93bed1d80626ae63762c5d64ab099e321c838beaa7ee24006ad50981a0a65ed",
    },
    ("PCG64", "seed", "inc", "variant", "cm-dxsm"): {
        "random_values": "eb406352a02f182cc65c5443d028e6ddd7af9260d6fccb9edda7b824d009e1df",
        "initial_state_hash": "73cf476d2aeb333f5f81a1095c2fba66f4e42223e4f32a36390aa83c91104b1e",
        "final_state_hash": "9cd88b41a398af2d2a98d1c0cdafce1181c42cbd0ca3e8ed2595d5b17733d217",
    },
    ("AESCounter", "seed"): {
        "random_values": "475b8b4363e7aee8bd5022f1a2a0a5a125ca701a3c7585396cc0c7bdfb56c35e",
        "initial_state_hash": "9437a0c633aac707ee481628600ced2f179aa0c02a6e60540cab412e4b80c22e",
        "final_state_hash": "e5006b5d9bb180b16c6dcb947126ad0431ec1433a2b42cab263fcaacd69400e6",
    },
    ("AESCounter", "key"): {
        "random_values": "a67bb222cd34bc6669fabcef696dcb59962b99f02c8d6f2973b2b7198c7b4f6e",
        "initial_state_hash": "3f010da816b9fb73c4302a7ae08ec827630408a646dfa154d332eaa54fc86831",
        "final_state_hash": "2bda54055d16f380b340b6afbdd934176d22c22ac02c3338f4124f200c07e90d",
    },
    ("AESCounter", "seed", "counter"): {
        "random_values": "90706c2682cfc6ddd5d7dd445b6e7d7576f637cc2893fe9223b84589291efa0a",
        "initial_state_hash": "896f63d51a7d1fbc58efcf3716eae3ab10f2d8b99d49e665dfe8dc7f4b4c5aa5",
        "final_state_hash": "4ee6b74e9f458a2f816113b998ce1c3c0635444e84b459f6e1d160cb3beae18f",
    },
    ("AESCounter", "key", "counter"): {
        "random_values": "db466f69593fddef03efdfa936b725025d671b7a007146feaae14db9f3ae590e",
        "initial_state_hash": "f575d8515d21c9849e37fa3655ce941737f8ed68da53577a1e677952ef85b55e",
        "final_state_hash": "7843875db631ba602965b95ee408b2f914182c5d5336c9c12dd55718649c53f9",
    },
    ("HC128", "seed"): {
        "random_values": "31d3cd1019ad913858bcdb1767fdf070c60db1b2610e017a522b27c1ae55adaf",
        "initial_state_hash": "555b19837a44f89419c2ec968bd17899d960d8252739011698238c99c0124cbd",
        "final_state_hash": "d1e9b91945af353faffbec551688c8a606532b4a46f47abad0317cf85967c2dd",
    },
    ("JSF", "seed"): {
        "random_values": "6fb1be3b54be2e910ec1ad90101032780fef8b1d0f8b0065c05829567975d786",
        "initial_state_hash": "a34cda995efe8c3a2ae3dfd12b7a8f2f776c2e830955de989c8fa472c4f4cb92",
        "final_state_hash": "22844959e7e6ce6892744378ae8e30c13a50eb9b54f18893b606bb9a92021cc3",
    },
    ("JSF", "seed", "seed_size", 1): {
        "random_values": "6fb1be3b54be2e910ec1ad90101032780fef8b1d0f8b0065c05829567975d786",
        "initial_state_hash": "a34cda995efe8c3a2ae3dfd12b7a8f2f776c2e830955de989c8fa472c4f4cb92",
        "final_state_hash": "22844959e7e6ce6892744378ae8e30c13a50eb9b54f18893b606bb9a92021cc3",
    },
    ("JSF", "seed", "seed_size", 2): {
        "random_values": "44d82d54c1d4840d40b4f51e1a35e4d5d6f345222ff1be50e1aa2164b2d8cfe8",
        "initial_state_hash": "2edc52764a6dee85a6b94ed22802d06d8b36b32fa4fde5020c880defc1dfd973",
        "final_state_hash": "8e781f3550a702116d688a1605df8cad5718269c892946342620b441de437072",
    },
    ("JSF", "seed", "seed_size", 3): {
        "random_values": "ee40c817d05a4ea5e73ccfe02512b57a315daa6ca1f51cd141caaf75d5c5b9fb",
        "initial_state_hash": "2da0db20b666b4603090048e30f845adbd5e6e254e92d2cced2d3e81f4cf76fa",
        "final_state_hash": "851c39dfacae62702acae90707213a7e417e39ca4a4aebbcad07bba8d2d1b20f",
    },
    ("JSF", "seed", "size", 32): {
        "random_values": "f77eea980fa123f6c5d4f23d54361e606fc0d947f5e95fa27867399980b903cf",
        "initial_state_hash": "ea34c93936af5ab55c3e7dfcc6d59d258c3b74f5580c6e0b496b0e8faa3d2d20",
        "final_state_hash": "39845780004254fba0b483bbf91adb4e8f01e75f97d37b25e10e018087895b5f",
    },
    ("JSF", "seed", "size", 64): {
        "random_values": "6fb1be3b54be2e910ec1ad90101032780fef8b1d0f8b0065c05829567975d786",
        "initial_state_hash": "a34cda995efe8c3a2ae3dfd12b7a8f2f776c2e830955de989c8fa472c4f4cb92",
        "final_state_hash": "22844959e7e6ce6892744378ae8e30c13a50eb9b54f18893b606bb9a92021cc3",
    },
    ("JSF", "seed", "seed_size", 1, "size", 32): {
        "random_values": "f77eea980fa123f6c5d4f23d54361e606fc0d947f5e95fa27867399980b903cf",
        "initial_state_hash": "ea34c93936af5ab55c3e7dfcc6d59d258c3b74f5580c6e0b496b0e8faa3d2d20",
        "final_state_hash": "39845780004254fba0b483bbf91adb4e8f01e75f97d37b25e10e018087895b5f",
    },
    ("JSF", "seed", "seed_size", 1, "size", 64): {
        "random_values": "6fb1be3b54be2e910ec1ad90101032780fef8b1d0f8b0065c05829567975d786",
        "initial_state_hash": "a34cda995efe8c3a2ae3dfd12b7a8f2f776c2e830955de989c8fa472c4f4cb92",
        "final_state_hash": "22844959e7e6ce6892744378ae8e30c13a50eb9b54f18893b606bb9a92021cc3",
    },
    ("JSF", "seed", "seed_size", 2, "size", 32): {
        "random_values": "bc2a4a226808516518c05ca5571bd07253d9d041cf8fc979ffeda8d53f2e2b00",
        "initial_state_hash": "20137cf86f159a9864b7e5cb7c113fc1d55aac4866802431cf491eb895a44200",
        "final_state_hash": "093e8a5ccf4db9cb8407cfc84921dd37bdb26874d19d768434f85e56b5c52122",
    },
    ("JSF", "seed", "seed_size", 2, "size", 64): {
        "random_values": "44d82d54c1d4840d40b4f51e1a35e4d5d6f345222ff1be50e1aa2164b2d8cfe8",
        "initial_state_hash": "2edc52764a6dee85a6b94ed22802d06d8b36b32fa4fde5020c880defc1dfd973",
        "final_state_hash": "8e781f3550a702116d688a1605df8cad5718269c892946342620b441de437072",
    },
    ("JSF", "seed", "seed_size", 3, "size", 32): {
        "random_values": "0ea49782e3a1b0761a56b1da457a630644411e799d12a29de503b2cc3782cef4",
        "initial_state_hash": "46ad4e252711e8bcbb3b28b2d4ff2c3bf9b1ce695538c99541c64e915cd56d52",
        "final_state_hash": "9b6de29739f0f887e7e754daf0dff425eb3ce9823a6bd74d0bf7c4862dde3d0f",
    },
    ("JSF", "seed", "seed_size", 3, "size", 64): {
        "random_values": "ee40c817d05a4ea5e73ccfe02512b57a315daa6ca1f51cd141caaf75d5c5b9fb",
        "initial_state_hash": "2da0db20b666b4603090048e30f845adbd5e6e254e92d2cced2d3e81f4cf76fa",
        "final_state_hash": "851c39dfacae62702acae90707213a7e417e39ca4a4aebbcad07bba8d2d1b20f",
    },
    ("Philox", "seed"): {
        "random_values": "bc90bdc019d1c5bac6bec53ca831379627592b0831b80b59e4afb263c5ce11fd",
        "initial_state_hash": "30761485a2e189c5742ebd975d220687573d5ab9bfbbf3839f911b94f32e9c19",
        "final_state_hash": "18a84797b828a86fb9e0cab427f91c1bc9953330bdc5886440cd111f1bfc6b4b",
    },
    ("Philox", "key"): {
        "random_values": "12db7bf022e0ab25a38e4cf0e42db3602a51f80d18012ee5db05701491bb9d00",
        "initial_state_hash": "d1f05f4db056a36b23a1b2253e1c7f93dc017cabca1e51a19a2bf65651e16e8b",
        "final_state_hash": "1397b47519ae57c06e16944b06aa2ea78e87cc3f9ed7ec1ce359401da75dc267",
    },
    ("Philox", "seed", "number", 2): {
        "random_values": "48e9b87b6f3dedcc0dea64d819a3769f2d359af7ed3d8c0e012ee00ebc0d61f4",
        "initial_state_hash": "952ceecddb435e1288518a966cdc1065ea995be57906ff813eb0610670eda77e",
        "final_state_hash": "73654fef7b61bc0812c8e47c8c8427169255fb519fce19598591f4fa4b381bf7",
    },
    ("Philox", "seed", "number", 4): {
        "random_values": "bc90bdc019d1c5bac6bec53ca831379627592b0831b80b59e4afb263c5ce11fd",
        "initial_state_hash": "30761485a2e189c5742ebd975d220687573d5ab9bfbbf3839f911b94f32e9c19",
        "final_state_hash": "18a84797b828a86fb9e0cab427f91c1bc9953330bdc5886440cd111f1bfc6b4b",
    },
    ("Philox", "seed", "width", 32): {
        "random_values": "33a57973b963d2c79ede4dfdc636a2ff89f5a681e9548ddc6480607072b4bcde",
        "initial_state_hash": "952ceecddb435e1288518a966cdc1065ea995be57906ff813eb0610670eda77e",
        "final_state_hash": "5213028a9f40eae78f2436535874b99baae91b4bcde8f4c7078a4a05be749973",
    },
    ("Philox", "seed", "width", 64): {
        "random_values": "bc90bdc019d1c5bac6bec53ca831379627592b0831b80b59e4afb263c5ce11fd",
        "initial_state_hash": "30761485a2e189c5742ebd975d220687573d5ab9bfbbf3839f911b94f32e9c19",
        "final_state_hash": "18a84797b828a86fb9e0cab427f91c1bc9953330bdc5886440cd111f1bfc6b4b",
    },
    ("Philox", "seed", "counter"): {
        "random_values": "20a8bffba45fb1e050e32dd65b19b6fca62b979c0452a23de872c479f2fe9e0c",
        "initial_state_hash": "ba545ca88e0b008ec765017768d33ce60b2f9abdb77abc456a4467c486e3d834",
        "final_state_hash": "619dfd078baf2f09adc48e5b48e3bbdc65e2e9fa162d6c0fa4cd39cdc3c7bc36",
    },
    ("Philox", "key", "number", 2): {
        "random_values": "2c77f61cb1d1c804d06d78918d6c8614ca2c1237f30b9950afcf6a58b74432eb",
        "initial_state_hash": "b63fbd1450adb24653bb23a4adbcb2768693cbe3f3ca651919ddf41d8e73de0a",
        "final_state_hash": "faa18fab243cc49176fb3530987c124c0a32cad610f4d7d266e20c9750c00374",
    },
    ("Philox", "key", "number", 4): {
        "random_values": "12db7bf022e0ab25a38e4cf0e42db3602a51f80d18012ee5db05701491bb9d00",
        "initial_state_hash": "d1f05f4db056a36b23a1b2253e1c7f93dc017cabca1e51a19a2bf65651e16e8b",
        "final_state_hash": "1397b47519ae57c06e16944b06aa2ea78e87cc3f9ed7ec1ce359401da75dc267",
    },
    ("Philox", "key", "width", 32): {
        "random_values": "370f7144658347110312b81b8fd91f650dcdbfcb72ced92899956e8d71653fd4",
        "initial_state_hash": "b63fbd1450adb24653bb23a4adbcb2768693cbe3f3ca651919ddf41d8e73de0a",
        "final_state_hash": "807565998082dcf4d9e9a95967add7b66ea8781f4fc065a863a9360a6fa9122e",
    },
    ("Philox", "key", "width", 64): {
        "random_values": "12db7bf022e0ab25a38e4cf0e42db3602a51f80d18012ee5db05701491bb9d00",
        "initial_state_hash": "d1f05f4db056a36b23a1b2253e1c7f93dc017cabca1e51a19a2bf65651e16e8b",
        "final_state_hash": "1397b47519ae57c06e16944b06aa2ea78e87cc3f9ed7ec1ce359401da75dc267",
    },
    ("Philox", "counter", "key"): {
        "random_values": "2d5252bfa2ef7d025757bed026eda0dfb771ceacb7c43e738f64e557e51b1fa3",
        "initial_state_hash": "fbfaa41dc1bb6a6907b804ba792a697dc4216a672a17484a261ffebc201ba450",
        "final_state_hash": "00cb74b3a8490396d98b524c273ad1893ca0e16288c350189385d9295bde8dec",
    },
    ("Philox", "seed", "number", 2, "width", 32): {
        "random_values": "e584623a8441d6ef640d757f444f827124f9712ac5b9d5185e48ce0dbfd6a82a",
        "initial_state_hash": "f19def0ef417f1cc91d7d6fb1ad9bf364fe3273da854c65c0ca3c032a94a9eb3",
        "final_state_hash": "8b561da675d9422966baaa7bc43d4a507f8861d28f9179345ce5f7ad2c17acf0",
    },
    ("Philox", "seed", "number", 2, "width", 64): {
        "random_values": "48e9b87b6f3dedcc0dea64d819a3769f2d359af7ed3d8c0e012ee00ebc0d61f4",
        "initial_state_hash": "952ceecddb435e1288518a966cdc1065ea995be57906ff813eb0610670eda77e",
        "final_state_hash": "73654fef7b61bc0812c8e47c8c8427169255fb519fce19598591f4fa4b381bf7",
    },
    ("Philox", "seed", "number", 4, "width", 32): {
        "random_values": "33a57973b963d2c79ede4dfdc636a2ff89f5a681e9548ddc6480607072b4bcde",
        "initial_state_hash": "952ceecddb435e1288518a966cdc1065ea995be57906ff813eb0610670eda77e",
        "final_state_hash": "5213028a9f40eae78f2436535874b99baae91b4bcde8f4c7078a4a05be749973",
    },
    ("Philox", "seed", "number", 4, "width", 64): {
        "random_values": "bc90bdc019d1c5bac6bec53ca831379627592b0831b80b59e4afb263c5ce11fd",
        "initial_state_hash": "30761485a2e189c5742ebd975d220687573d5ab9bfbbf3839f911b94f32e9c19",
        "final_state_hash": "18a84797b828a86fb9e0cab427f91c1bc9953330bdc5886440cd111f1bfc6b4b",
    },
    ("Philox", "seed", "counter", "number", 2): {
        "random_values": "770e71df438e88a4caecc5e1136f06b375eb9360bd2d3edb3888b2daf1bdbf41",
        "initial_state_hash": "0d7fdf509d611392fca89d1e335e96b502df9d5a120922b5d33fbe5284759379",
        "final_state_hash": "e64c30fb05984cb8c9860f426137eaf6cae40654e19323d499ac420869e94f8d",
    },
    ("Philox", "seed", "counter", "number", 4): {
        "random_values": "20a8bffba45fb1e050e32dd65b19b6fca62b979c0452a23de872c479f2fe9e0c",
        "initial_state_hash": "ba545ca88e0b008ec765017768d33ce60b2f9abdb77abc456a4467c486e3d834",
        "final_state_hash": "619dfd078baf2f09adc48e5b48e3bbdc65e2e9fa162d6c0fa4cd39cdc3c7bc36",
    },
    ("Philox", "seed", "counter", "width", 32): {
        "random_values": "6a542bd74cd64d0edd390505df5438d3d5dce6fb26842803e8742a876cf699b8",
        "initial_state_hash": "0d7fdf509d611392fca89d1e335e96b502df9d5a120922b5d33fbe5284759379",
        "final_state_hash": "b9e14a513841ca06404f29984867c2021e5d2fba5700f09de76efcb620edcb32",
    },
    ("Philox", "seed", "counter", "width", 64): {
        "random_values": "20a8bffba45fb1e050e32dd65b19b6fca62b979c0452a23de872c479f2fe9e0c",
        "initial_state_hash": "ba545ca88e0b008ec765017768d33ce60b2f9abdb77abc456a4467c486e3d834",
        "final_state_hash": "619dfd078baf2f09adc48e5b48e3bbdc65e2e9fa162d6c0fa4cd39cdc3c7bc36",
    },
    ("Philox", "key", "number", 2, "width", 32): {
        "random_values": "fd75f7908fe6226864714fa239ba21e36cd9a420c849f075ebbf513b42f54336",
        "initial_state_hash": "16fc92755c50e72e4406ea52412541c33360f468fe1828663716ad59d0ecb440",
        "final_state_hash": "5027e50d7960c014e1043be6812a0ccfdec1bb50c81bd824a7477e01d9a6b64e",
    },
    ("Philox", "key", "number", 2, "width", 64): {
        "random_values": "2c77f61cb1d1c804d06d78918d6c8614ca2c1237f30b9950afcf6a58b74432eb",
        "initial_state_hash": "b63fbd1450adb24653bb23a4adbcb2768693cbe3f3ca651919ddf41d8e73de0a",
        "final_state_hash": "faa18fab243cc49176fb3530987c124c0a32cad610f4d7d266e20c9750c00374",
    },
    ("Philox", "key", "number", 4, "width", 32): {
        "random_values": "370f7144658347110312b81b8fd91f650dcdbfcb72ced92899956e8d71653fd4",
        "initial_state_hash": "b63fbd1450adb24653bb23a4adbcb2768693cbe3f3ca651919ddf41d8e73de0a",
        "final_state_hash": "807565998082dcf4d9e9a95967add7b66ea8781f4fc065a863a9360a6fa9122e",
    },
    ("Philox", "key", "number", 4, "width", 64): {
        "random_values": "12db7bf022e0ab25a38e4cf0e42db3602a51f80d18012ee5db05701491bb9d00",
        "initial_state_hash": "d1f05f4db056a36b23a1b2253e1c7f93dc017cabca1e51a19a2bf65651e16e8b",
        "final_state_hash": "1397b47519ae57c06e16944b06aa2ea78e87cc3f9ed7ec1ce359401da75dc267",
    },
    ("Philox", "counter", "key", "number", 2): {
        "random_values": "e2ee5c58e58b7684de3791a3d925023d4b281602d2880b7a6594cf46e0e0974e",
        "initial_state_hash": "4c618a9c433ba3b2cebc0279f5f465e065bfa0adcffec15ef91e6981f342ba98",
        "final_state_hash": "8ad25108731a3f801cb8fc1b892b5e36948135b97386c273ff0fd73f3217267d",
    },
    ("Philox", "counter", "key", "number", 4): {
        "random_values": "2d5252bfa2ef7d025757bed026eda0dfb771ceacb7c43e738f64e557e51b1fa3",
        "initial_state_hash": "fbfaa41dc1bb6a6907b804ba792a697dc4216a672a17484a261ffebc201ba450",
        "final_state_hash": "00cb74b3a8490396d98b524c273ad1893ca0e16288c350189385d9295bde8dec",
    },
    ("Philox", "counter", "key", "width", 32): {
        "random_values": "b95b97b641dcd4998a4b341b6f19add14e02e5aae9f8dc3b5013d90b0e4853e6",
        "initial_state_hash": "4c618a9c433ba3b2cebc0279f5f465e065bfa0adcffec15ef91e6981f342ba98",
        "final_state_hash": "078b3dabc40e9bd2691b8568a681a69c1f0635783d3061e73cc87ae77c6739f8",
    },
    ("Philox", "counter", "key", "width", 64): {
        "random_values": "2d5252bfa2ef7d025757bed026eda0dfb771ceacb7c43e738f64e557e51b1fa3",
        "initial_state_hash": "fbfaa41dc1bb6a6907b804ba792a697dc4216a672a17484a261ffebc201ba450",
        "final_state_hash": "00cb74b3a8490396d98b524c273ad1893ca0e16288c350189385d9295bde8dec",
    },
    ("Philox", "seed", "counter", "number", 2, "width", 32): {
        "random_values": "8f02206bc4e20a4d6f0e0b74382a65269edb62ba390cdf716330f1f7d2c53a1b",
        "initial_state_hash": "ef680cafbf049f064693808656c1b3aad7e9089257e54f11e27179c545bec60b",
        "final_state_hash": "a807386ed676eb3710b0e87be34205a44389f7a49b21d5b9f37b91bab953096f",
    },
    ("Philox", "seed", "counter", "number", 2, "width", 64): {
        "random_values": "770e71df438e88a4caecc5e1136f06b375eb9360bd2d3edb3888b2daf1bdbf41",
        "initial_state_hash": "0d7fdf509d611392fca89d1e335e96b502df9d5a120922b5d33fbe5284759379",
        "final_state_hash": "e64c30fb05984cb8c9860f426137eaf6cae40654e19323d499ac420869e94f8d",
    },
    ("Philox", "seed", "counter", "number", 4, "width", 32): {
        "random_values": "6a542bd74cd64d0edd390505df5438d3d5dce6fb26842803e8742a876cf699b8",
        "initial_state_hash": "0d7fdf509d611392fca89d1e335e96b502df9d5a120922b5d33fbe5284759379",
        "final_state_hash": "b9e14a513841ca06404f29984867c2021e5d2fba5700f09de76efcb620edcb32",
    },
    ("Philox", "seed", "counter", "number", 4, "width", 64): {
        "random_values": "20a8bffba45fb1e050e32dd65b19b6fca62b979c0452a23de872c479f2fe9e0c",
        "initial_state_hash": "ba545ca88e0b008ec765017768d33ce60b2f9abdb77abc456a4467c486e3d834",
        "final_state_hash": "619dfd078baf2f09adc48e5b48e3bbdc65e2e9fa162d6c0fa4cd39cdc3c7bc36",
    },
    ("Philox", "counter", "key", "number", 2, "width", 32): {
        "random_values": "4b52ff53a5a9816b7c4021c3f5c67624cd391539af94b71bf0d5d3772e600aa6",
        "initial_state_hash": "e6b0370aabea8d7b675a597b194d9019dc924e66748d64bba4b7c1ad0207f05a",
        "final_state_hash": "be4f7acb5ab3b6fc2a0ba4d3e974a9dbf44d48e0da3e3b61e3be2dc1dd707190",
    },
    ("Philox", "counter", "key", "number", 2, "width", 64): {
        "random_values": "e2ee5c58e58b7684de3791a3d925023d4b281602d2880b7a6594cf46e0e0974e",
        "initial_state_hash": "4c618a9c433ba3b2cebc0279f5f465e065bfa0adcffec15ef91e6981f342ba98",
        "final_state_hash": "8ad25108731a3f801cb8fc1b892b5e36948135b97386c273ff0fd73f3217267d",
    },
    ("Philox", "counter", "key", "number", 4, "width", 32): {
        "random_values": "b95b97b641dcd4998a4b341b6f19add14e02e5aae9f8dc3b5013d90b0e4853e6",
        "initial_state_hash": "4c618a9c433ba3b2cebc0279f5f465e065bfa0adcffec15ef91e6981f342ba98",
        "final_state_hash": "078b3dabc40e9bd2691b8568a681a69c1f0635783d3061e73cc87ae77c6739f8",
    },
    ("Philox", "counter", "key", "number", 4, "width", 64): {
        "random_values": "2d5252bfa2ef7d025757bed026eda0dfb771ceacb7c43e738f64e557e51b1fa3",
        "initial_state_hash": "fbfaa41dc1bb6a6907b804ba792a697dc4216a672a17484a261ffebc201ba450",
        "final_state_hash": "00cb74b3a8490396d98b524c273ad1893ca0e16288c350189385d9295bde8dec",
    },
    ("ThreeFry", "seed"): {
        "random_values": "6a0ce1a57c1d7fdc91efcc519176b7f8fb7553d684ff07038a659862fbb877e9",
        "initial_state_hash": "3ec541383acb10f44f514e66bd8f88d4cc3a7f44e21833b16cb8e99320c35d0c",
        "final_state_hash": "3c1216fd3b3593b5aa542088c2e22b2872178d0bbfd9eff6ed0e511e822666aa",
    },
    ("ThreeFry", "key"): {
        "random_values": "a8ea53a7e3a559890a91356fc31ae879f30a3ea88baf14f7997faa8a23894457",
        "initial_state_hash": "986fbf9d513f8a13ca75ef09af64c5c7a54a6e4e50a5eee46409832d3194d806",
        "final_state_hash": "4a4aa5c96a65f20af12d0506277a289bf82b867b95d69acb6b13a0a4e2302a8e",
    },
    ("ThreeFry", "seed", "number", 2): {
        "random_values": "5cf840b4f12d5bcbee80fa0fb01969b13421923bbf06eb69c5417f01ea14365b",
        "initial_state_hash": "7211b0124e6809ac81ea1b8492aea65aa6d06fb50ca8bdb6c9c6d348ddea59bd",
        "final_state_hash": "bc9193ec6a19404662f435b22fead3e90d9708219d2aaa7f588cf2de6ad4aa0b",
    },
    ("ThreeFry", "seed", "number", 4): {
        "random_values": "6a0ce1a57c1d7fdc91efcc519176b7f8fb7553d684ff07038a659862fbb877e9",
        "initial_state_hash": "3ec541383acb10f44f514e66bd8f88d4cc3a7f44e21833b16cb8e99320c35d0c",
        "final_state_hash": "3c1216fd3b3593b5aa542088c2e22b2872178d0bbfd9eff6ed0e511e822666aa",
    },
    ("ThreeFry", "seed", "width", 32): {
        "random_values": "04933a58593c142d9b9acd8d3e220b44b45f68ca4e756e5c305bdace792e0d2f",
        "initial_state_hash": "7211b0124e6809ac81ea1b8492aea65aa6d06fb50ca8bdb6c9c6d348ddea59bd",
        "final_state_hash": "55d11d158d9bb8de344c2b579f29d1bcc0a6b5a579364194b635dbbe78676b84",
    },
    ("ThreeFry", "seed", "width", 64): {
        "random_values": "6a0ce1a57c1d7fdc91efcc519176b7f8fb7553d684ff07038a659862fbb877e9",
        "initial_state_hash": "3ec541383acb10f44f514e66bd8f88d4cc3a7f44e21833b16cb8e99320c35d0c",
        "final_state_hash": "3c1216fd3b3593b5aa542088c2e22b2872178d0bbfd9eff6ed0e511e822666aa",
    },
    ("ThreeFry", "seed", "counter"): {
        "random_values": "c6f1520b128c4af38bf3d94ea0644364da3d10686392f1636bf069a6b8da2735",
        "initial_state_hash": "970672b5c24558658568db365f052cc6aefb1e55494f66d40f24a375f9e5826a",
        "final_state_hash": "564afaebfcd1a77c15bcf38e74f883c92d306160373ca521a1ae06057dbc54a6",
    },
    ("ThreeFry", "key", "number", 2): {
        "random_values": "33cbaedfe895ec6fe5349b2f90ae3adda09e349f33bdc02121644a6558221603",
        "initial_state_hash": "1aa0f6c54afdcfa3e40e3a5b7879d94805a2e89d0d021ba2224288e44952c8e2",
        "final_state_hash": "8519ef938885a112834bb5b3e8025b6ed024f5b943f0eebf34f6fbda347b078c",
    },
    ("ThreeFry", "key", "number", 4): {
        "random_values": "a8ea53a7e3a559890a91356fc31ae879f30a3ea88baf14f7997faa8a23894457",
        "initial_state_hash": "986fbf9d513f8a13ca75ef09af64c5c7a54a6e4e50a5eee46409832d3194d806",
        "final_state_hash": "4a4aa5c96a65f20af12d0506277a289bf82b867b95d69acb6b13a0a4e2302a8e",
    },
    ("ThreeFry", "key", "width", 32): {
        "random_values": "2ba849e210f225b6ff5ce29e44149e5e7e0cc97bd0199d713259479305104adc",
        "initial_state_hash": "1aa0f6c54afdcfa3e40e3a5b7879d94805a2e89d0d021ba2224288e44952c8e2",
        "final_state_hash": "4af303c32bbbd8053aa879223d47d17cc2afbb07e256e467a727e002972d8c47",
    },
    ("ThreeFry", "key", "width", 64): {
        "random_values": "a8ea53a7e3a559890a91356fc31ae879f30a3ea88baf14f7997faa8a23894457",
        "initial_state_hash": "986fbf9d513f8a13ca75ef09af64c5c7a54a6e4e50a5eee46409832d3194d806",
        "final_state_hash": "4a4aa5c96a65f20af12d0506277a289bf82b867b95d69acb6b13a0a4e2302a8e",
    },
    ("ThreeFry", "counter", "key"): {
        "random_values": "25b866f7ea1c07c038a7c9171600818aa56f301eba78b78b99e18f12e63be3c3",
        "initial_state_hash": "e4e02e7fa750d132dbbb4eca5aa478f5e2145db7c9cbaee2189a6ce2bfeb4bc0",
        "final_state_hash": "12907fe1ea0da104af1a3a19043c6da7dbb29b3f6e688adfe4bbea22e9066bdc",
    },
    ("ThreeFry", "seed", "number", 2, "width", 32): {
        "random_values": "a4864e23a19857ec1f9e01c190d9173c1c6f015ae8420754870b64fb0dd26bc9",
        "initial_state_hash": "1bb00b26a5a55001809bcefa3e5388802ee9863eb8d5cf66a2dc86b1a49680c5",
        "final_state_hash": "4a11dadb31c66cf93d0030af6502a040cb710cb115360d301b5d29085958a1ae",
    },
    ("ThreeFry", "seed", "number", 2, "width", 64): {
        "random_values": "5cf840b4f12d5bcbee80fa0fb01969b13421923bbf06eb69c5417f01ea14365b",
        "initial_state_hash": "7211b0124e6809ac81ea1b8492aea65aa6d06fb50ca8bdb6c9c6d348ddea59bd",
        "final_state_hash": "bc9193ec6a19404662f435b22fead3e90d9708219d2aaa7f588cf2de6ad4aa0b",
    },
    ("ThreeFry", "seed", "number", 4, "width", 32): {
        "random_values": "04933a58593c142d9b9acd8d3e220b44b45f68ca4e756e5c305bdace792e0d2f",
        "initial_state_hash": "7211b0124e6809ac81ea1b8492aea65aa6d06fb50ca8bdb6c9c6d348ddea59bd",
        "final_state_hash": "55d11d158d9bb8de344c2b579f29d1bcc0a6b5a579364194b635dbbe78676b84",
    },
    ("ThreeFry", "seed", "number", 4, "width", 64): {
        "random_values": "6a0ce1a57c1d7fdc91efcc519176b7f8fb7553d684ff07038a659862fbb877e9",
        "initial_state_hash": "3ec541383acb10f44f514e66bd8f88d4cc3a7f44e21833b16cb8e99320c35d0c",
        "final_state_hash": "3c1216fd3b3593b5aa542088c2e22b2872178d0bbfd9eff6ed0e511e822666aa",
    },
    ("ThreeFry", "seed", "counter", "number", 2): {
        "random_values": "c6bf820bad3fc39ac51b7da69cee0a0a9be74fbfdcf717231f8ec0035ab73620",
        "initial_state_hash": "7d20a5553e1ad0a7d48974fcafad9a31febdcfc8faff2e1cd56574b6095e68d1",
        "final_state_hash": "24161a97a4f1820f5800d16b235fc01e3c80a7662b4440f281059a3fd62b54b8",
    },
    ("ThreeFry", "seed", "counter", "number", 4): {
        "random_values": "c6f1520b128c4af38bf3d94ea0644364da3d10686392f1636bf069a6b8da2735",
        "initial_state_hash": "970672b5c24558658568db365f052cc6aefb1e55494f66d40f24a375f9e5826a",
        "final_state_hash": "564afaebfcd1a77c15bcf38e74f883c92d306160373ca521a1ae06057dbc54a6",
    },
    ("ThreeFry", "seed", "counter", "width", 32): {
        "random_values": "764e93381248f0ff0f440460cceff9ae1ac1bbacba3619490684b620c8265b77",
        "initial_state_hash": "7d20a5553e1ad0a7d48974fcafad9a31febdcfc8faff2e1cd56574b6095e68d1",
        "final_state_hash": "78fca6e00b00f218f43fc80ec4df11674164d5f9fca2932c534f74c37d007125",
    },
    ("ThreeFry", "seed", "counter", "width", 64): {
        "random_values": "c6f1520b128c4af38bf3d94ea0644364da3d10686392f1636bf069a6b8da2735",
        "initial_state_hash": "970672b5c24558658568db365f052cc6aefb1e55494f66d40f24a375f9e5826a",
        "final_state_hash": "564afaebfcd1a77c15bcf38e74f883c92d306160373ca521a1ae06057dbc54a6",
    },
    ("ThreeFry", "key", "number", 2, "width", 32): {
        "random_values": "c757a77a4c2dc6710cac3c2b031e4d832fb6862fd1d744fff95d7baef78f6c3e",
        "initial_state_hash": "8ac1c50d9d20e4e45c2048f18ae8371baba4f49a3450113bad95f48bbc4781c2",
        "final_state_hash": "bae81cafd31eb0a9900041652596b5fe9397fdb66f33f648b95db1e07af22e10",
    },
    ("ThreeFry", "key", "number", 2, "width", 64): {
        "random_values": "33cbaedfe895ec6fe5349b2f90ae3adda09e349f33bdc02121644a6558221603",
        "initial_state_hash": "1aa0f6c54afdcfa3e40e3a5b7879d94805a2e89d0d021ba2224288e44952c8e2",
        "final_state_hash": "8519ef938885a112834bb5b3e8025b6ed024f5b943f0eebf34f6fbda347b078c",
    },
    ("ThreeFry", "key", "number", 4, "width", 32): {
        "random_values": "2ba849e210f225b6ff5ce29e44149e5e7e0cc97bd0199d713259479305104adc",
        "initial_state_hash": "1aa0f6c54afdcfa3e40e3a5b7879d94805a2e89d0d021ba2224288e44952c8e2",
        "final_state_hash": "4af303c32bbbd8053aa879223d47d17cc2afbb07e256e467a727e002972d8c47",
    },
    ("ThreeFry", "key", "number", 4, "width", 64): {
        "random_values": "a8ea53a7e3a559890a91356fc31ae879f30a3ea88baf14f7997faa8a23894457",
        "initial_state_hash": "986fbf9d513f8a13ca75ef09af64c5c7a54a6e4e50a5eee46409832d3194d806",
        "final_state_hash": "4a4aa5c96a65f20af12d0506277a289bf82b867b95d69acb6b13a0a4e2302a8e",
    },
    ("ThreeFry", "counter", "key", "number", 2): {
        "random_values": "0fb5d5f947d1d66624ae4b20bf3948b3fbe5cf3678d7e5dd2ecdb11312859bc2",
        "initial_state_hash": "607f8291746c6df525905f15668aaad47cbed77fb6429e558e591d37c9355571",
        "final_state_hash": "d707ee9ae02e137529dda83bd487ef3774c7d8217cc7abaa73c0f6e0741a761a",
    },
    ("ThreeFry", "counter", "key", "number", 4): {
        "random_values": "25b866f7ea1c07c038a7c9171600818aa56f301eba78b78b99e18f12e63be3c3",
        "initial_state_hash": "e4e02e7fa750d132dbbb4eca5aa478f5e2145db7c9cbaee2189a6ce2bfeb4bc0",
        "final_state_hash": "12907fe1ea0da104af1a3a19043c6da7dbb29b3f6e688adfe4bbea22e9066bdc",
    },
    ("ThreeFry", "counter", "key", "width", 32): {
        "random_values": "f80b442b0be3b37c27fe45beae10ead34adb3f8da63af4277911c2d13b24a6c7",
        "initial_state_hash": "607f8291746c6df525905f15668aaad47cbed77fb6429e558e591d37c9355571",
        "final_state_hash": "8272b56782c9d0dae1a3fa2a7f8a7cb0bfd53187a38064ace03ddbe98f2172d4",
    },
    ("ThreeFry", "counter", "key", "width", 64): {
        "random_values": "25b866f7ea1c07c038a7c9171600818aa56f301eba78b78b99e18f12e63be3c3",
        "initial_state_hash": "e4e02e7fa750d132dbbb4eca5aa478f5e2145db7c9cbaee2189a6ce2bfeb4bc0",
        "final_state_hash": "12907fe1ea0da104af1a3a19043c6da7dbb29b3f6e688adfe4bbea22e9066bdc",
    },
    ("ThreeFry", "seed", "counter", "number", 2, "width", 32): {
        "random_values": "7ab2f3577b70421764dea0c632b977e6baa359a3245b9fcc62b0a881f22babf8",
        "initial_state_hash": "d981173fa32b4608bc8647680a8a247f20564ecb317b6bc69d88c175262d5da6",
        "final_state_hash": "59e777cbd3292f44dd997a1059a6555aa327371cd549ead3c55504170728463a",
    },
    ("ThreeFry", "seed", "counter", "number", 2, "width", 64): {
        "random_values": "c6bf820bad3fc39ac51b7da69cee0a0a9be74fbfdcf717231f8ec0035ab73620",
        "initial_state_hash": "7d20a5553e1ad0a7d48974fcafad9a31febdcfc8faff2e1cd56574b6095e68d1",
        "final_state_hash": "24161a97a4f1820f5800d16b235fc01e3c80a7662b4440f281059a3fd62b54b8",
    },
    ("ThreeFry", "seed", "counter", "number", 4, "width", 32): {
        "random_values": "764e93381248f0ff0f440460cceff9ae1ac1bbacba3619490684b620c8265b77",
        "initial_state_hash": "7d20a5553e1ad0a7d48974fcafad9a31febdcfc8faff2e1cd56574b6095e68d1",
        "final_state_hash": "78fca6e00b00f218f43fc80ec4df11674164d5f9fca2932c534f74c37d007125",
    },
    ("ThreeFry", "seed", "counter", "number", 4, "width", 64): {
        "random_values": "c6f1520b128c4af38bf3d94ea0644364da3d10686392f1636bf069a6b8da2735",
        "initial_state_hash": "970672b5c24558658568db365f052cc6aefb1e55494f66d40f24a375f9e5826a",
        "final_state_hash": "564afaebfcd1a77c15bcf38e74f883c92d306160373ca521a1ae06057dbc54a6",
    },
    ("ThreeFry", "counter", "key", "number", 2, "width", 32): {
        "random_values": "1c092eeb065ee79faeb4608402b65cbc00242c643aceb504b581302618d59348",
        "initial_state_hash": "1ebd47d99ce2f9c44b4e21abe6c2ff7be502f16078bab05c4ddd04f1eb036f61",
        "final_state_hash": "64d2c6dee5afdf2b6124e09805680910ef868ecef4bf774536b7393a049a2dc7",
    },
    ("ThreeFry", "counter", "key", "number", 2, "width", 64): {
        "random_values": "0fb5d5f947d1d66624ae4b20bf3948b3fbe5cf3678d7e5dd2ecdb11312859bc2",
        "initial_state_hash": "607f8291746c6df525905f15668aaad47cbed77fb6429e558e591d37c9355571",
        "final_state_hash": "d707ee9ae02e137529dda83bd487ef3774c7d8217cc7abaa73c0f6e0741a761a",
    },
    ("ThreeFry", "counter", "key", "number", 4, "width", 32): {
        "random_values": "f80b442b0be3b37c27fe45beae10ead34adb3f8da63af4277911c2d13b24a6c7",
        "initial_state_hash": "607f8291746c6df525905f15668aaad47cbed77fb6429e558e591d37c9355571",
        "final_state_hash": "8272b56782c9d0dae1a3fa2a7f8a7cb0bfd53187a38064ace03ddbe98f2172d4",
    },
    ("ThreeFry", "counter", "key", "number", 4, "width", 64): {
        "random_values": "25b866f7ea1c07c038a7c9171600818aa56f301eba78b78b99e18f12e63be3c3",
        "initial_state_hash": "e4e02e7fa750d132dbbb4eca5aa478f5e2145db7c9cbaee2189a6ce2bfeb4bc0",
        "final_state_hash": "12907fe1ea0da104af1a3a19043c6da7dbb29b3f6e688adfe4bbea22e9066bdc",
    },
}
